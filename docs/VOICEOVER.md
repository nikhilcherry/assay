# Voiceover for the demo video

For `demo.mp4` (3:02, release `replay-v1`). Read at an easy pace, about 140 words
a minute; each segment fits its time with room to breathe. The captions stay on
screen, so the voice doesn't need to repeat numbers; it says what they mean.

To lay it over the video once recorded (`voice.m4a`):

```bash
ffmpeg -i demo.mp4 -i voice.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest demo-voiced.mp4
```

---

**0:00 Title** (~15 words)
> This is assay: a fraud investigator that works on a graph, and shows its reasoning as it goes.

**0:08 The hook** (~20 words)
> Here's a seventy-five dollar online purchase. The bank's own fraud model looked at it and said: almost certainly fine.

**0:14 Walking the graph** (~35 words)
> Watch what the agent does. It asks what card this is, what that person normally buys, what phone they used, and then the question the bank's model never asks: who else used that exact phone?

**0:30 The ring** (~30 words)
> Twenty-eight different cards. Sixty purchases. One handset, hiding behind an anonymous connection. None of those purchases looked wrong on its own. You only see it when you connect them.

**0:42 The decision** (~20 words)
> So it blocks the card, puts the other twenty-seven under watch, and writes the report a regulator would expect.

**0:50 Checking itself** (~25 words)
> And you can check its reasoning. Take away the phone clue, and the same case drops back to "probably fine." The connection was the evidence.

**0:58 Structuring** (~30 words)
> This one is four purchases, each just under five hundred dollars, in half an hour. The bank's list of fraud types doesn't cover it. Its own case history does, and the agent finds it there.

**1:30 An innocent customer** (~30 words)
> Now the opposite. The bank's model was worried about this purchase. The agent looked at how this person always spends, and let it through. Half these cases were innocent people.

**1:48 Not sure** (~25 words)
> Sometimes the evidence genuinely doesn't settle it. Instead of guessing, the agent says so, asks for more information, and hands it to a person.

**2:05 A correction** (~35 words)
> This one I got wrong this morning. A customer said "not me," and the model said "fine." I checked the bank's history: every single dispute it investigated was fraud. So now a low score can't overrule a customer.

**2:25 Going looking** (~25 words)
> Then it went looking on its own, and opened sixty investigations nobody asked for. It even remembers its own earlier verdicts.

**2:40 The big picture** (~20 words)
> Every dot is the agent; every tick is the bank. The lines show where they disagree.

**2:48 Why trust it** (~30 words)
> Why trust the numbers? Because they're tested on a month the model never saw. When it says thirty percent, about three in ten turn out to be fraud.

**2:55 Close** (~15 words)
> assay. It connects the dots, shows its work, and leaves innocent customers alone.
