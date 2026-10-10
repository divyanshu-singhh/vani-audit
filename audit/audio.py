"""Audio probing via the ffmpeg binary bundled with imageio-ffmpeg (no system install)."""
import re
import subprocess

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> dict:
    """Return {duration_sec, channels, sample_rate}; values are None if unreadable."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True, errors="replace").stderr
    info = {"duration_sec": None, "channels": None, "sample_rate": None}
    if m := re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out):
        h, mnt, s = m.groups()
        info["duration_sec"] = int(h) * 3600 + int(mnt) * 60 + float(s)
    if m := re.search(r"Audio: .*?(\d+) Hz, (mono|stereo|\d+ channels)", out):
        info["sample_rate"] = int(m.group(1))
        ch = m.group(2)
        info["channels"] = 1 if ch == "mono" else 2 if ch == "stereo" else int(ch.split()[0])
    return info
