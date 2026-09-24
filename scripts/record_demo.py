"""Record the demo video from the replay page, unattended.

    python -m http.server 8731 -d site &
    python scripts/record_demo.py            # -> docs/demo.mp4

Needs `pip install playwright`, ffmpeg, and a Chrome/Chromium binary
(CHROME=/path, default /usr/bin/google-chrome). Frames come from the DevTools
screencast (sharp text, unlike a VP8 recording) and are stitched with ffmpeg at
their real timestamps. The captions are the narration; every number they quote
is on screen at the time.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("ASSAY_URL", "http://127.0.0.1:8731/")
CHROME = os.environ.get("CHROME", "/usr/bin/google-chrome")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "demo.mp4"
W, H = 1440, 900

OVERLAY = """
(() => {
  const s = document.createElement('style');
  s.textContent = `
  #rec-cap{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);max-width:1100px;width:calc(100% - 80px);
    background:rgba(14,19,23,.93);color:#E6EAE6;font:500 21px/1.4 "Familjen Grotesk",Arial,sans-serif;padding:14px 22px;
    z-index:99;transition:opacity .4s}
  #rec-cap b{color:#F0674E;font-weight:700} #rec-cap i{color:#D7A650;font-style:normal} #rec-cap u{color:#5BC2A6;text-decoration:none}
  #rec-card{position:fixed;inset:0;background:#0E1317;color:#E6EAE6;z-index:100;display:flex;flex-direction:column;
    justify-content:center;padding:0 140px;gap:22px;transition:opacity .6s}
  #rec-card h1{font:700 76px/1 "Familjen Grotesk",Arial,sans-serif;letter-spacing:-.03em;margin:0}
  #rec-card p{font:400 26px/1.45 "Source Serif 4",Georgia,serif;color:#A9B3BB;margin:0;max-width:60ch}
  #rec-card code{font:500 20px/1.4 "JetBrains Mono",monospace;color:#7D93FF}`;
  document.head.appendChild(s);
  const c = document.createElement('div'); c.id = 'rec-cap'; c.style.opacity = 0; document.body.appendChild(c);
  window.cap = (html) => { c.style.opacity = html ? 1 : 0; if (html) c.innerHTML = html; };
  window.card = (html) => { let k = document.getElementById('rec-card');
    if (!html) { if (k) { k.style.opacity = 0; setTimeout(() => k.remove(), 700); } return; }
    if (!k) { k = document.createElement('div'); k.id = 'rec-card'; document.body.appendChild(k); }
    k.innerHTML = html; k.style.opacity = 1; };
})();
"""


async def main() -> int:
    frames_dir = Path(tempfile.mkdtemp(prefix="assay-demo-"))
    frames: list[tuple[float, Path]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--hide-scrollbars"])
        page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="light")
        await page.goto(URL + "#HHG-014")
        await page.wait_for_function("window.assayReplay && document.querySelector('#verdict').textContent.length > 0")
        await page.evaluate(OVERLAY)
        await page.wait_for_timeout(1500)

        cdp = await page.context.new_cdp_session(page)

        async def on_frame(ev):
            path = frames_dir / f"{len(frames):06d}.jpg"
            frames.append((ev["metadata"]["timestamp"], path))
            path.write_bytes(base64.b64decode(ev["data"]))
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:
                pass
        cdp.on("Page.screencastFrame", lambda ev: asyncio.ensure_future(on_frame(ev)))
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H})

        ev, wait = page.evaluate, page.wait_for_timeout

        async def until_done():
            await wait(800)
            await page.wait_for_function("assayReplay.done", timeout=180_000, polling=200)

        async def replay(case_id, speed=1.0, tab=None, hold=1800):
            if tab:
                await page.click(tab)
            await ev(f"assayReplay.setSpeed({speed})")
            await ev(f"assayReplay.load('{case_id}', {{autoplay:true}})")
            await until_done()
            await wait(hold)

        async def scroll_to(sel, offset=0):
            await ev(f"""(() => {{ const y = document.querySelector('{sel}').getBoundingClientRect().top + scrollY - {offset};
                window.scrollTo({{top: y, behavior: 'smooth'}}); }})()""")
            await wait(1400)

        # title
        await ev("""card(`<h1>assay</h1><p>An agentic fraud investigator on TigerGraph, with probabilities that are
          measured rather than guessed.</p><p>Every frame that follows is a real run of the agent, replayed query by query.
          No language model sits on the decision path: 0 tokens.</p><code>Hacker House Goa 2026 · Task 4</code>`)""")
        await wait(6500)
        await ev("card(null)")
        await wait(900)
        await ev("cap(`HHG-014: an analyst flagged one <i>$74.96</i> online purchase. The bank's model scored it <i>0.05</i>.`)")
        await wait(5000)

        # the ring
        await scroll_to("#bench", 8)
        await ev("cap(`Watch the agent walk the graph: transaction → card → device → <b>every other card that device touched</b>.`)")
        await ev("assayReplay.setSpeed(0.9)")
        await ev("assayReplay.load('HHG-014', {autoplay:true})")
        await wait(9500)
        await ev("cap(`One Samsung handset behind an anonymous proxy, marked New on every card: <b>60 purchases on 28 cards</b>. Likelihood ratio ×30.`)")
        await until_done()
        await ev("cap(`Blocked, the 27 connected cards put under monitoring, and a report filed under R6 and R9. <u>10 queries in 0.02 s.</u>`)")
        await wait(2500)
        await page.click("details.sar summary")
        await wait(5500)

        # structuring
        await ev("cap(`HHG-006: four purchases just under <b>$500</b> in 30 minutes. None of the five documented patterns covers it. The bank's closed cases do.`)")
        await replay("HHG-006", 1.3, hold=3000)

        # a legitimate one
        await ev("cap(`HHG-017: the bank scored it <i>0.57</i>. The transaction fits the holder's own history, so it's <u>cleared</u>. Half the cases are legitimate, and blocking everything scores badly.`)")
        await replay("HHG-017", 1.3, hold=2500)

        # uncertain
        await ev("cap(`HHG-002: at <i>0.55</i> the evidence doesn't settle it. It asks for evidence, doesn't invent an answer, and holds the case for a human.`)")
        await replay("HHG-002", 1.3, hold=3000)

        # monitor + case memory
        await ev("cap(`Then it went looking on its own: <b>60 investigations nobody asked for</b>. The bank scored MON-041 at 0.03.`)")
        await replay("MON-041", 1.4, tab="#tab-mon", hold=500)
        await ev("cap(`Case memory: its earlier conclusion on HHG-014 is now a vertex in the graph, and it multiplies these odds by 3.`)")
        await wait(5500)

        # the two charts
        await ev("cap(null)")
        await scroll_to("#period", 30)
        await ev("cap(`All 80 investigations. Each dot is assay's probability, each brass tick is the bank's score, and each line shows where they disagree.`)")
        await wait(7500)
        await scroll_to("#calibration", 30)
        await ev("cap(`Why trust the number? The model learns only from the bank's closed cases and is tested on a month it never saw: <u>AUC 0.964 vs 0.866</u>. When it says 0.30, three in ten were fraud.`)")
        await wait(8500)
        await ev("cap(null)")
        await ev("""card(`<h1>assay</h1><p>8 fraud · 11 legitimate · 1 held for a human, across 20 cases.<br>60 more that it opened on its own.</p>
          <code>github.com/nikhilcherry/assay</code><code>nikhilcherry.github.io/assay</code>`)""")
        await wait(6000)

        await cdp.send("Page.stopScreencast")
        await wait(300)
        await browser.close()

    if len(frames) < 10:
        print("no frames captured", file=sys.stderr)
        return 1
    # The screencast sends a frame only when the page changes; hold each one for its real duration.
    lines = []
    for (t, f), (t2, _) in zip(frames, frames[1:] + [(frames[-1][0] + 2.0, None)]):
        lines += [f"file '{f}'", f"duration {max(0.001, t2 - t):.4f}"]
    lines.append(f"file '{frames[-1][1]}'")
    lst = frames_dir / "frames.txt"
    lst.write_text("\n".join(lines) + "\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-vf", f"fps=30,scale={W}:{H}:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-movflags", "+faststart", str(OUT)], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"{OUT}: {len(frames)} frames, {frames[-1][0] - frames[0][0]:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
