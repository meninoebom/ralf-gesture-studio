# Research Journal

Design observations, questions, and connections as they come up during development. Not polished writing — working notes for future reference.

---

## 2026-02-04 — What I'm actually building toward

The thing I keep coming back to is that Afro-diasporic choreomusical systems encode design grammars that Western HCI hasn't engaged with seriously. Call-and-response, polyrhythmic coordination, collective entrainment — these aren't just cultural practices, they're interaction paradigms. And they're almost entirely non-screen.

That's the core of what I'm focused on: **interaction that isn't mediated by a screen.** Audio feedback, rhythmic cues, felt timing. The screen is a fallback, not the primary channel. RALF is the first proof-of-concept — a gesture training system where the dancer never needs to look at a display — but the design grammar applies much more broadly. Rehabilitation, elder care, education, any context where someone's attention should be in their body, not on a screen.

Future direction: multi-body interaction. These traditions are fundamentally collective. A solo dancer training gestures is a starting point, but the real design space opens up when you have multiple bodies coordinating through sound and movement, not through shared screens.

## 2026-02-04 — Call-and-response as training structure

The training session in RALF uses a countdown → capture → rest cycle. I designed it this way because the dancer needs to know when to move without looking at the screen. The countdown beeps are the "call," the dancer's movement is the "response."

This isn't a metaphor — it's literally how training works in West African and Afro-diasporic dance traditions. The drummer plays a signal phrase, the dancer responds with the movement. The system knows when to expect movement because it initiated the call. The dancer knows when to move because they heard the call. No screen needed.

I didn't fully realize this connection when I first implemented it. I was solving a practical UX problem (how do you train a classifier when the user can't look at a screen?) and arrived at a structure that turns out to be indigenous to the practice itself. That's worth paying attention to — it suggests the design grammar isn't being imposed from outside, it's being recovered.

## 2026-02-04 — Audio over visual: a deliberate inversion

Most ML training interfaces are screen-first. You watch a visualization, click buttons, review results on a dashboard. RALF inverts this: the primary feedback channel during training is audio (countdown ticks, capture start beep, completion ding). The screen shows state for reference, but the system is designed to work even if you never look at it.

This came from watching dancers use Wekinator. They'd record a gesture, then break their physical state to walk over and check the screen, then try to get back into their body to record the next one. The context switch was killing their ability to produce consistent, high-quality examples. The training data suffered because the tool's interaction model conflicted with the practitioner's mode of engagement.

The insight: **the feedback modality has to match the practitioner's attentional mode.** Dancers attend with their ears and bodies, not their eyes. A tool that demands visual attention is asking them to leave the very state that produces good training data.

This principle generalizes. Rehab patients doing physical therapy shouldn't have to watch a screen. Elders doing movement exercises shouldn't need to interpret visual dashboards. The audio-first principle isn't specific to dance — it's specific to any embodied practice.

## 2026-02-04 — The warm-up effect as a design question

There's a measurable warm-up effect in the first 3-5 seconds of a recognition session. The sliding window needs real movement data before DTW distances become meaningful. First hits in a session average ~34% margin vs ~68% for established ones.

From an engineering perspective, this is a known limitation of windowed approaches. But from a design perspective, it mirrors something real about embodied practice: you don't start cold. Dancers warm up. Musicians tune. There's a preparation phase before the system (human or technical) is ready to perform.

Should the system acknowledge this explicitly? A "tuning" phase where the recognizer openly says "I'm not ready yet"? Or does that break the metaphor — in call-and-response, the drummer doesn't announce they're warming up, they just start with simpler patterns. Maybe the answer is a graduated entry rather than a hard threshold. Question to sit with.

## 2026-02-04 — Coordinate systems and perspective

Today I built a gesture viewer that visualizes recorded skeleton data as a stick figure. The figure showed up rotated 90 degrees — the tracking system's x-axis pointed down (head to feet) instead of left to right. A simple axis swap fixed it, but it surfaced something worth noting.

Movement data doesn't have an inherent orientation. The coordinate system is a choice made by the tracking pipeline, and it's invisible until something goes wrong. Whose perspective does the coordinate system encode? The camera's? The dancer's? The audience's? MediaPipe normalizes to camera-view, which means the dancer sees themselves mirrored. That's a colonial flattening in miniature — the observer's frame of reference is treated as default.

Not sure what to do with this yet. But the question of *whose perspective is encoded in the data representation* feels connected to the larger project.

---

*To add an entry: date header, short title, write what's on your mind. Tag nothing. Let patterns emerge.*
