# main.py
import os
import tempfile
import subprocess
import numpy as np
import yt_dlp
import pysrt
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import google.generativeai as genai
import cv2
import imageio_ffmpeg
import moviepy.config as mpy_config

# --- Paths & Colors ---
FONT_PATH = "ui/font.ttf"
BG_IMAGE = "ui/bg_template.jpg"
WIDTH, HEIGHT = 720, 1280
INVESHO_BLUE = "#4285F4"
WHITE_COLOR = "#FFFFFF"

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe")
mpy_config.change_settings({"FFMPEG_BINARY": FFMPEG_PATH})

# --- Load API Keys ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Helpers ---
def auto_crop(video_clip):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    frame = video_clip.get_frame(0)
    cv2.imwrite(tmp, frame)
    gray = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(thresh != 0)
    return video_clip.crop(x1=xs.min(), y1=ys.min(), x2=xs.max(), y2=ys.max())

def transcribe_video_and_get_text(video_path, max_duration=None):
    import whisper
    if not os.path.exists(FFPROBE_PATH):
        raise RuntimeError("ffprobe not found. Cannot check audio.")
    result = subprocess.run([FFPROBE_PATH, "-i", video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"], capture_output=True, text=True)
    if not result.stdout.strip():
        raise RuntimeError("No audio found. Please upload a video with sound.")

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
            f.write(f"{i+1}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")
            full_transcript.append(text)
    return " ".join(full_transcript)

def format_srt_time(seconds):
    import datetime
    t = datetime.timedelta(seconds=seconds)
    s = str(t)
    s = f"{s.split('.')[0]},{str(t.microseconds)[:3]}" if "." in s else f"{s},000"
    h, m, s = s.split(":")
    return f"{int(h):02}:{int(m):02}:{s}"

def generate_short_quote(transcript):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        prompt = f"Transcript: '''{transcript}'''\nReturn a short 3-5 word quote:"
        return model.generate_content(prompt).text.strip().replace('"', '')
    except Exception:
        return "Key Insights"

def download_instagram_video(insta_url):
    import re, uuid
    match = re.search(r"/reel/([\w\-]+)", insta_url)
    if not match:
        raise ValueError("Invalid Instagram URL")
    reel_id = match.group(1)
    out_path = f"downloads/video_{uuid.uuid4().hex[:8]}.%(ext)s"
    cookie_path = os.getenv("IG_COOKIE_PATH", "cookies.txt")
    ydl_opts = {
        "ffmpeg_location": FFMPEG_PATH,
        "ffprobe_location": FFPROBE_PATH,
        "format": "bestvideo+bestaudio/best",
        "outtmpl": out_path,
        "cookiefile": cookie_path,
        "quiet": True,
        "no_warnings": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.instagram.com/reel/{reel_id}/", download=True)
        return ydl.prepare_filename(info)

def generate_subtitles(video, srt_path):
    subs = pysrt.open(srt_path)
    font = ImageFont.truetype(FONT_PATH, 24)
    clips = []
    for sub in subs:
        start = sub.start.ordinal / 1000
        duration = sub.duration.seconds + sub.duration.milliseconds / 1000.0
        text = sub.text.replace("\n", " ")
        img = Image.new("RGBA", (video.w, 40), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((20, 5), text, font=font, fill=WHITE_COLOR)
        clips.append(ImageClip(np.array(img)).set_duration(duration).set_start(start).set_position(("center", video.h - 50)))
    return clips

def render_stacked_header(title, quote, size, duration):
    title_font = ImageFont.truetype(FONT_PATH, 50)
    quote_font = ImageFont.truetype(FONT_PATH, 30)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((40, 20), title, font=title_font, fill=INVESHO_BLUE)
    draw.text((40, 90), quote, font=quote_font, fill=WHITE_COLOR)
    return ImageClip(np.array(img)).set_duration(duration)

def render_branding_text(duration):
    font1 = ImageFont.truetype(FONT_PATH, 50)
    font2 = ImageFont.truetype(FONT_PATH, 20)
    img = Image.new("RGBA", (500, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "Invesho", font=font1, fill=INVESHO_BLUE)
    draw.text((0, 60), "Accelerating access to", font=font2, fill=WHITE_COLOR)
    draw.text((0, 85), "startup capital.", font=font2, fill=WHITE_COLOR)
    return ImageClip(np.array(img)).set_duration(duration).set_position(("right", "bottom")).margin(right=60, bottom=80)

def create_final_video(video_path, title_text, quote_text, max_duration):
    clip = VideoFileClip(video_path).subclip(0, max_duration)
    cropped = auto_crop(clip)
    video = cropped.resize(width=WIDTH - 120).set_position(("center", "center"))
    bg = ImageClip(BG_IMAGE).resize((WIDTH, HEIGHT)).set_duration(video.duration)
    mask = Image.new("L", video.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, video.size[0], video.size[1]], radius=40, fill=255)
    video = video.set_mask(ImageClip(np.array(mask) / 255).set_duration(video.duration).set_ismask(True))
    header = render_stacked_header(title_text, quote_text, (WIDTH, 200), video.duration).set_position(("center", "top")).margin(top=80)
    branding = render_branding_text(video.duration)
    layers = [bg, video, header, branding]
    if os.path.exists("final.srt"):
        layers.extend(generate_subtitles(video, "final.srt"))
    final = CompositeVideoClip(layers)
    final.write_videofile("final_with_subs.mp4", fps=24, codec="libx264")

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Invesho Generator", layout="centered")
    st.title("📽️ Invesho Instagram Reel Generator")

    col1, col2 = st.columns(2)
    with col1:
        insta_url = st.text_input("📎 Instagram Reel URL")
    with col2:
        uploaded_file = st.file_uploader("📤 Or upload a video (MP4/MOV/WEBM)", type=["mp4", "mov", "webm"])

    title_text = st.text_input("Enter Title", "Mark Zuckerberg")
    video_duration = st.slider("Max Video Duration (seconds)", 5, 60, 20)

    if st.button("🎬 Generate Video"):
        if not (insta_url or uploaded_file) or not title_text:
            st.warning("Upload a video or URL and add title.")
            return

        with st.spinner("Step 1/4: Getting video..."):
            try:
                if uploaded_file:
                    os.makedirs("downloads", exist_ok=True)
                    video_path = os.path.join("downloads", uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_instagram_video(insta_url)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return

        try:
            with st.spinner("Step 2/4: Transcribing..."):
                transcript = transcribe_video_and_get_text(video_path, video_duration)
        except Exception as e:
            st.error(str(e))
            return

        with st.spinner("Step 3/4: Generating quote..."):
            quote = generate_short_quote(transcript)

        with st.spinner("Step 4/4: Creating final video..."):
            create_final_video(video_path, title_text, quote, video_duration)

        st.success("✅ Done!")
        st.video("final_with_subs.mp4")
        with open("final_with_subs.mp4", "rb") as f:
            st.download_button("⬇️ Download", f, "invesho_reel.mp4")

if __name__ == "__main__":
    main()
