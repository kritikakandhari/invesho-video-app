# main.py
import os
import re
import uuid
import numpy as np
import tempfile
import subprocess
import datetime

import whisper
import soundfile as sf
import pysrt
import cv2
import imageio_ffmpeg
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from moviepy.editor import (
    VideoFileClip, CompositeVideoClip, ImageClip, AudioFileClip
)
import google.generativeai as genai
import yt_dlp
import moviepy.config as mpy_config

# --- Constants ---
FONT_PATH = "ui/font.ttf"
BG_IMAGE = "ui/bg_template.jpg"
WIDTH, HEIGHT = 720, 1280
INVESHO_BLUE = "#4285F4"
WHITE_COLOR = "#FFFFFF"

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
mpy_config.change_settings({"FFMPEG_BINARY": FFMPEG_PATH})

# --- Load API Keys ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# --- Auto-Crop ---
def auto_crop(video_clip):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    frame = video_clip.get_frame(0)
    cv2.imwrite(tmp, frame)
    gray = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(thresh != 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return video_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


# --- Transcription ---
def transcribe_video_and_get_text(video_path, max_duration=None):
    try:
        audio = AudioFileClip(video_path)
        audio.close()
    except Exception:
        raise RuntimeError("This video has no audio track. Please upload a video with sound.")

    def patched_load_audio(file: str, sr: int = 16000):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        cmd = [
            FFMPEG_PATH, "-y", "-i", file,
            "-ac", "1", "-ar", str(sr),
            "-f", "wav", wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.strip()}")

        audio, _ = sf.read(wav_path, dtype="float32")
        return whisper.audio.pad_or_trim(audio)

    whisper.audio.load_audio = patched_load_audio
    model = whisper.load_model("small")
    result = model.transcribe(video_path, language="en")

    full_transcript = []
    with open("final.srt", "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            if max_duration and segment["start"] > max_duration:
                break
            start = segment["start"]
            end = min(segment["end"], max_duration) if max_duration else segment["end"]
            text = segment["text"].strip()
            f.write(f"{i+1}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")
            full_transcript.append(text)

    return " ".join(full_transcript)


# --- Time Formatter ---
def format_srt_time(seconds):
    t = str(datetime.timedelta(seconds=seconds))
    if "." in t:
        s, ms = t.split(".")
        return f"{s},{ms[:3]}"
    return f"{t},000"


# --- Gemini Quote Generator ---
def generate_short_quote(transcript):
    if not transcript:
        return "Key Insights"
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        prompt = f"""
        Based on the following transcript, generate a single, very short quote or tagline (3 to 5 words) that captures the main idea.
        Transcript: '''{transcript}'''
        Return ONLY the short quote.
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', '')
    except Exception:
        return "Key Video Insights"


# --- Instagram Downloader ---
def download_instagram_video(insta_url):
    import re
    import uuid

    # Get path to cookies.txt from .env
    cookie_path = os.getenv("IG_COOKIE_PATH")

    if not cookie_path or not os.path.exists(cookie_path):
        raise FileNotFoundError("cookies.txt not found or IG_COOKIE_PATH is not set correctly in .env")

    match = re.search(r"(?:https?://)?(?:www\.)?instagram\.com/reel/([a-zA-Z0-9_\-]+)/?", insta_url)
    if not match:
        raise ValueError("Invalid Instagram Reel URL")

    reel_id = match.group(1)
    clean_url = f"https://www.instagram.com/reel/{reel_id}/"
    unique_id = str(uuid.uuid4())[:8]
    os.makedirs("downloads", exist_ok=True)

    ydl_opts = {
        "ffmpeg_location": FFMPEG_PATH,
        "format": "bestvideo+bestaudio/best",
        "outtmpl": f"downloads/video_{unique_id}.%(ext)s",
        "cookiefile": cookie_path,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept-Language": "en-US,en;q=0.9",
        },
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "cachedir": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(clean_url, download=True)
            return ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(f"Failed to download video: {e}")



# --- Subtitle Generation ---
def generate_subtitles(video, srt_path, video_top_y):
    subs = pysrt.open(srt_path)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()

    subtitle_clips = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        duration = sub.duration.ordinal / 1000.0
        text = sub.text.replace("\n", " ")
        dummy = Image.new("RGBA", (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2], bbox[3]
        img = Image.new("RGBA", (w + 10, h + 6), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((5, 3), text, font=font, fill=(255, 255, 255, 255))

        subtitle = (
            ImageClip(np.array(img))
            .set_duration(duration)
            .set_start(start)
            .set_position(("center", video_top_y + 10))
        )
        subtitle_clips.append(subtitle)
    return subtitle_clips


# --- UI Rendering ---
def render_stacked_header(title, quote, size, duration):
    title_font = ImageFont.truetype(FONT_PATH, 50)
    quote_font = ImageFont.truetype(FONT_PATH, 30)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    quote_bbox = draw.textbbox((0, 0), quote, font=quote_font)
    draw.text(((size[0] - title_bbox[2]) / 2, (size[1] / 2) - title_bbox[3]), title, font=title_font, fill=INVESHO_BLUE)
    draw.text(((size[0] - quote_bbox[2]) / 2, (size[1] / 2) + 10), quote, font=quote_font, fill=WHITE_COLOR)
    return ImageClip(np.array(img)).set_duration(duration)


def render_branding_text(duration):
    brand_font = ImageFont.truetype(FONT_PATH, 50)
    tagline_font = ImageFont.truetype(FONT_PATH, 18)

    # Create branding image
    img = Image.new("RGBA", (500, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "Invesho", font=brand_font, fill=INVESHO_BLUE)
    draw.text((0, 60), "Accelerating access to", font=tagline_font, fill=WHITE_COLOR)
    draw.text((0, 85), "startup capital.", font=tagline_font, fill=WHITE_COLOR)

    branding_clip = ImageClip(np.array(img)).set_duration(duration)

    # ❗ Force precise pixel location (bottom-right with padding)
    padding_right = 40
    padding_bottom = 40
    x_pos = WIDTH - 500 - padding_right
    y_pos = HEIGHT - 120 - padding_bottom

    return branding_clip.set_position((x_pos, y_pos))



# --- Final Video Composer ---
def create_final_video(video_path, title_text, quote_text, max_duration, use_subs):
    raw_clip = VideoFileClip(video_path)
    duration = min(max_duration, raw_clip.duration)
    raw = raw_clip.subclip(0, duration)
    cropped = auto_crop(raw)
    video = cropped.resize(width=WIDTH - 40).set_position(("center", "center"))

    mask = Image.new("L", video.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, video.size[0], video.size[1]], radius=40, fill=255)
    video = video.set_mask(ImageClip(np.array(mask) / 255).set_duration(video.duration).set_ismask(True))

    bg = ImageClip(BG_IMAGE).resize((WIDTH, HEIGHT)).set_duration(video.duration)
    header = render_stacked_header(title_text, quote_text, (WIDTH, 200), video.duration).set_position(("center", "top"))
    branding = render_branding_text(video.duration)

    layers = [bg, video, header, branding]
    video_top_y = (HEIGHT - video.size[1]) // 2

    if use_subs and os.path.exists("final.srt"):
        layers.extend(generate_subtitles(video, "final.srt", video_top_y))

    final = CompositeVideoClip(layers, size=(WIDTH, HEIGHT))
    final.write_videofile("final_with_subs.mp4", fps=24, codec="libx264")


# --- Streamlit Interface ---
def main():
    st.set_page_config(page_title="Invesho Insta Generator", layout="centered")
    st.title("Invesho Instagram Reel Generator")

    st.markdown("### 1. Provide a Video")
    col1, col2 = st.columns(2)
    with col1:
        insta_url = st.text_input("Instagram Reel URL")
    with col2:
        uploaded_file = st.file_uploader("Or Upload a video (MP4/MOV/WEBM)", type=["mp4", "mov", "webm"])

    title_text_input = st.text_input("2. Enter Main Title", "Mark Zuckerberg")
    st.subheader("Settings")
    video_duration = st.slider("Max Video Duration (seconds)", 5, 60, 20)
    use_subs = st.checkbox("Add In-Video Subtitles", value=True)
    st.divider()

    if st.button("Generate Video", type="primary"):
        if not (insta_url or uploaded_file) or not title_text_input:
            st.warning("Please provide a video file or Instagram URL and a title.")
            return

        try:
            with st.spinner("Step 1/4: Getting video..."):
                if uploaded_file:
                    path = os.path.join("downloads", uploaded_file.name)
                    os.makedirs("downloads", exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.read())
                    video_path = path
                else:
                    video_path = download_instagram_video(insta_url)
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return

        try:
            with st.spinner("Step 2/4: Transcribing video..."):
                transcript = transcribe_video_and_get_text(video_path, max_duration=video_duration)
        except RuntimeError as e:
            st.error(str(e))
            return

        with st.spinner("Step 3/4: Generating AI quote..."):
            quote = generate_short_quote(transcript)

        with st.spinner("Step 4/4: Creating final video..."):
            create_final_video(video_path, title_text_input, quote, video_duration, use_subs)

        st.success("Done! Watch below:")
        st.video("final_with_subs.mp4")
        with open("final_with_subs.mp4", "rb") as file:
            st.download_button("Download Video", file, "invesho_reel.mp4")


if __name__ == "__main__":
    main()
