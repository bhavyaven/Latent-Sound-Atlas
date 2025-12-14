// Scholarly Project: Latent Sound Atlas
// Bhavya Venkataraghavan

(
s.waitForBoot({
	// =========================================================================
	// REFINED SYNTHDEFS
	// =========================================================================

	// 1. CELTIC HARP
	SynthDef(\celticHarp, {
		arg out=0, freq=440, pan=0, amp=2.5, dur=0.5, decayArg=5, verbMix=0.3;
		var sig, env;
		sig = Pluck.ar(
			in: PinkNoise.ar(0.1),
			trig: Impulse.kr(0),
			maxdelaytime: 1/50,
			delaytime: 1 / (freq + LFNoise1.kr(2).range(-1, 1)),
			decaytime: decayArg,
			coef: 0.5
		);
		sig = LPF.ar(sig, 2000);
		env = EnvGen.kr(Env.perc(0.005, dur), doneAction: 2);
		sig = sig * env * amp * 0.8;
		sig = Pan2.ar(sig, pan);
		sig = FreeVerb.ar(sig, 0.5, 0.9, 0.3) * verbMix + sig * (1 - verbMix);
		Out.ar(out, sig);
	}).writeDefFile;

	// 2. WHIMSICAL FLUTE
	SynthDef(\forestFlute, {
		arg out=0, freq=440, pan=0, amp=0.2, dur=0.5, breathLevel=0.3, cutoff=3000;
		var sig, env, breath, vibrato;
		env = EnvGen.kr(Env.perc(0.05, dur), doneAction: 2);
		vibrato = SinOsc.kr(5, 0, freq * 0.005) * Line.kr(0, 1, 1.0);
		breath = PinkNoise.ar(breathLevel);
		breath = HPF.ar(breath, 2000) * env;
		sig = SinOsc.ar(freq + vibrato);
		sig = Pan2.ar(sig, pan) + (Pan2.ar(breath, 0) * 0.5);
		sig = LPF.ar(sig, cutoff);
		sig = FreeVerb.ar(sig, 0.4, 0.8, 0.1);
		Out.ar(out, sig);
	}).writeDefFile;

	// 3. FAIRY SPARKLES
	SynthDef(\fairySparkle, {
		arg out=0, freq=2000, pan=0, amp=0.3, dur=5, density=1;
		var sig, env, sparkles;
		env = EnvGen.kr(Env.linen(0.1, dur-0.2, 0.1), doneAction: 2);
		sparkles = Dust.ar(density);
		sig = Ringz.ar(sparkles, [freq, freq*1.4, freq*2.1, freq*(2.5 + LFNoise1.kr(0.1).range(0.5, 1.3))], 0.8).mean;
		sig = HPF.ar(sig, 800);
		sig = sig * env * amp;
		sig = Limiter.ar(sig, 0.7);
		sig = Pan2.ar(sig, pan);
		sig = FreeVerb.ar(sig, 0.6, 0.95, 0.1);
		Out.ar(out, sig);
	}).writeDefFile;

	// 3B. FAIRY CHIME
	SynthDef(\fairyChime, {
		arg out=0, freq=2000, pan=0, amp=0.5, dur=4, density=20, verbMix=0.5;
		var sig, env, sparkles;
		sparkles = Dust.ar(Line.kr(density, 1, dur));
		sig = Ringz.ar(sparkles, [freq, freq*1.51, freq*2.42, freq*3.23, freq*5.1], 0.5);
		sig = Mix(sig);
		env = EnvGen.kr(Env.linen(0.05, dur, 0.5, 0.7), doneAction: 2);
		sig = HPF.ar(sig, 1000);
		sig = sig * env * amp * 0.5;
		sig = Pan2.ar(sig, pan);
		sig = FreeVerb.ar(sig, 0.5, 0.95, 0.1) * verbMix + sig * (1 - verbMix);
		Out.ar(out, sig);
	}).writeDefFile;

	// 4. MYSTIC PAD
	SynthDef(\mysticPad, {
		arg out=0, freq=110, pan=0, amp=0.4, dur=6, cutoffBase=500, cutoffRange=1000;
		var sig, env, mod;
		env = EnvGen.kr(Env.sine(dur), doneAction: 2);
		sig = VarSaw.ar([freq, freq*1.01, freq*0.995], 0, 0.3).mean;
		mod = SinOsc.kr(0.1).range(cutoffBase, cutoffBase + cutoffRange);
		sig = RLPF.ar(sig, mod, 0.6);
		sig = sig * env * amp;
		sig = Pan2.ar(sig, pan);
		Out.ar(out, sig);
	}).writeDefFile;

	// 5. BABBLING BROOK
	SynthDef(\babblingBrook, {
		arg out=0, amp=5.0, dur=10, filterFreq=14, bubbleGain=0.2;
		var sig, env, excite, lowBubbles, highBubbles;
		env = EnvGen.kr(Env.linen(0.5, dur - 0.55, 0.75), doneAction: 2);
		excite = OnePole.ar(BrownNoise.ar(1 ! 2), 0.99);
		lowBubbles = RHPF.ar(excite, LPF.ar(BrownNoise.ar(1 ! 2), filterFreq) * 400 + 500, 0.03, 0.003);
		highBubbles = RHPF.ar(excite, LPF.ar(BrownNoise.ar(1 ! 2), filterFreq + 6) * 800 + 1000, 0.03, 0.005);
		sig = (lowBubbles + (highBubbles * 0.5)) * bubbleGain * env;
		sig = sig * amp;
		sig = Limiter.ar(sig, 0.9);
		Out.ar(out, sig);
	}).writeDefFile;

	// 6. GENTLE RAIN
	SynthDef(\relaxingRain, {
		arg out=0, amp=0.2, dur=10, brownMix=0.3, cutoffLow=300;
		var sig, env;
		env = EnvGen.kr(Env.linen(1, dur-2, 1), doneAction: 2);
		sig = BrownNoise.ar(brownMix) + PinkNoise.ar(1 - brownMix);
		sig = LPF.ar(sig, 1500);
		sig = LPF.ar(sig, cutoffLow);
		sig = sig * env * amp;
		sig = Pan2.ar(sig, LFNoise1.kr(0.1).range(-0.4, 0.4));
		Out.ar(out, sig);
	}).writeDefFile;

	// 7. SPELL WHOOSH
	SynthDef(\softWhoosh, {
		arg out=0, amp=0.5, dur=2, sweepStart=200, sweepEnd=3000;
		var sig, env, sweep;
		env = EnvGen.kr(Env.sine(dur), doneAction: 2);
		sig = PinkNoise.ar(1);
		sweep = EnvGen.kr(Env([sweepStart, sweepEnd, sweepStart], [dur/2, dur/2], \exp));
		sig = BPF.ar(sig, sweep, 0.4);
		sig = sig * env * amp;
		sig = Pan2.ar(sig, Line.kr(-1, 1, dur));
		Out.ar(out, sig);
	}).writeDefFile;

	// 8. OCARINA
	SynthDef(\ocarinaNote, {
		arg out=0, freq=523, pan=0, amp=0.5, dur=0.5, vibRate=4, vibDepth=0.005;
		var sig, env, vibrato;
		env = EnvGen.kr(Env.perc(0.05, dur), doneAction: 2);
		vibrato = SinOsc.kr(vibRate, 0, freq * vibDepth);
		sig = SinOsc.ar(freq + vibrato);
		sig = sig + (WhiteNoise.ar(0.05) * EnvGen.kr(Env.perc(0.01, 0.05)));
		sig = sig * env * amp;
		sig = Pan2.ar(sig, pan);
		sig = FreeVerb.ar(sig, 0.4, 0.7, 0.1);
		Out.ar(out, sig);
	}).writeDefFile;

	// 9. MORNING BIRD
	SynthDef(\morningBird, {
		arg out=0, amp=0.3, pan=0, freqA=1200, freqB=2000;
		var sig, env, freqEnv, rate;
		env = EnvGen.kr(Env.perc(0.01, 0.2), doneAction: 2);
		freqEnv = EnvGen.ar(Env([freqA, freqB, freqA/2], [0.1, 0.1], \exp), 1, doneAction: 0);
		rate = SinOsc.ar(200);
		sig = rate * SinOsc.ar(freqEnv);
		sig = LPF.ar(sig, 4000);
		sig = sig * env * amp;
		sig = Pan2.ar(sig, pan);
		Out.ar(out, sig);
	}).writeDefFile;

	// 10. OWL HOOT
	SynthDef(\owlHoot, {
		arg out=0, amp=0.5, pan=0, freq=500, tremoloRate=6;
		var sig, env, tremolo;
		env = EnvGen.kr(Env.new([0, 1, 0.2, 0.8, 0], [0.1, 0.2, 0.2, 0.6], -4), doneAction: 2);
		tremolo = SinOsc.kr(tremoloRate).range(0.8, 1.0);
		sig = SinOsc.ar(freq);
		sig = sig * env * tremolo * amp;
		sig = LPF.ar(sig, 800);
		sig = Pan2.ar(sig, pan);
		sig = FreeVerb.ar(sig, 0.5, 0.9, 0.2);
		Out.ar(out, sig);
	}).writeDefFile;

	// 11. FIRE CRACKLING
	SynthDef(\fireCrackle, {
		arg out=0, amp=0.3, dur=8, density=8, lowPass=300;
		var sig, env, rumble, snaps;
		env = EnvGen.kr(Env.linen(0.5, dur-1, 0.5), doneAction: 2);
		rumble = BrownNoise.ar(0.4);
		rumble = LPF.ar(rumble, lowPass);
		snaps = Dust.ar(density);
		snaps = Decay.ar(snaps, 0.03) * PinkNoise.ar(0.5);
		snaps = HPF.ar(snaps, 1500);
		sig = Mix([rumble, snaps * 0.5]);
		sig = sig * env * amp;
		sig = Pan2.ar(sig, 0);
		Out.ar(out, sig);
	}).writeDefFile;

	// 12. RUSTLING LEAVES
	SynthDef(\rustlingLeaves, {
		arg out=0, amp=0.4, dur=6, modRate=1.0, filterCenter=1400;
		var sig, env, mod;
		env = EnvGen.kr(Env.sine(dur), doneAction: 2);
		sig = PinkNoise.ar(1);
		mod = LFNoise1.kr(modRate).range(0.2, 1.0);
		sig = BPF.ar(sig, LFNoise1.kr(0.5).range(filterCenter - 600, filterCenter + 600), 0.5);
		sig = sig * mod * env * amp;
		sig = Pan2.ar(sig, LFNoise1.kr(0.1));
		Out.ar(out, sig);
	}).writeDefFile;

	"All Fantasy SynthDefs Updated and Parameterized.".postln;

	s.sync;

	// =========================================================================
	// OSC BRIDGE - CONNECT TO OPENGL PROGRAM
	// =========================================================================
	OSCdef(\fantasyPlay, { |msg, time, addr, recvPort|
		var soundType, note, amp;

		// Log raw message
		"=== OSC Message Received ===".postln;
		"Address: %".format(msg[0]).postln;
		"Args: %".format(msg[1..]).postln;

		soundType = msg[1].asSymbol;
		note = msg[2] ? 440;
		amp = msg[3] ? 0.5;

		"Playing: type=%, freq=%, amp=%".format(soundType, note, amp).postln;

		case

		// ========== MELODIC INSTRUMENTS ==========
		{ soundType == \harp } {
			Synth(\celticHarp, [
				\freq, note,
				\amp, amp * 2.5,  // Harps need volume boost
				\dur, 3.0,
				\decayArg, rrand(4.0, 7.0),
				\verbMix, 0.5
			]);
			"→ Played celticHarp".postln;
		}

		{ soundType == \flute } {
			Synth(\forestFlute, [
				\freq, note,
				\amp, amp * 0.4,  // Flutes are naturally louder
				\dur, 2.5,
				\breathLevel, 0.3,
				\cutoff, 3000
			]);
			"→ Played forestFlute".postln;
		}

		{ soundType == \ocarina } {
			Synth(\ocarinaNote, [
				\freq, note,
				\amp, amp,
				\dur, 2.0,
				\vibRate, 5,
				\vibDepth, 0.005
			]);
			"→ Played ocarinaNote".postln;
		}

		{ soundType == \pad } {
			Synth(\mysticPad, [
				\freq, note * 0.5,  // Lower octave for pads
				\amp, amp * 0.8,
				\dur, 6.0,  // Longer duration
				\cutoffBase, 500,
				\cutoffRange, 1000
			]);
			"→ Played mysticPad".postln;
		}

		// ========== MAGIC EFFECTS ==========
		{ soundType == \sparkle } {
			Synth(\fairySparkle, [
				\freq, note,
				\amp, amp * 0.6,
				\dur, 5.0,
				\density, rrand(10, 20)
			]);
			"→ Played fairySparkle".postln;
		}

		{ soundType == \chime } {
			Synth(\fairyChime, [
				\freq, note,
				\amp, amp,
				\dur, 4.0,
				\density, 25,
				\verbMix, 0.6
			]);
			"→ Played fairyChime".postln;
		}

		// ========== NATURE AMBIENCES ==========
		{ soundType == \brook } {
			Synth(\babblingBrook, [
				\amp, amp * 5.0,  // Brook needs significant boost
				\dur, 8.0,
				\filterFreq, 14,
				\bubbleGain, 0.2
			]);
			"→ Played babblingBrook".postln;
		}

		{ soundType == \rain } {
			Synth(\relaxingRain, [
				\amp, amp * 0.4,
				\dur, 10.0,
				\brownMix, 0.3,
				\cutoffLow, 300
			]);
			"→ Played relaxingRain".postln;
		}

		{ soundType == \fire } {
			Synth(\fireCrackle, [
				\amp, amp * 0.6,
				\dur, 8.0,
				\density, 8,
				\lowPass, 300
			]);
			"→ Played fireCrackle".postln;
		}

		{ soundType == \leaves } {
			Synth(\rustlingLeaves, [
				\amp, amp * 0.8,
				\dur, 6.0,
				\modRate, 1.0,
				\filterCenter, 1400
			]);
			"→ Played rustlingLeaves".postln;
		}

		// ========== SOUND EFFECTS ==========
		{ soundType == \whoosh } {
			Synth(\softWhoosh, [
				\amp, amp,
				\dur, 2.0,
				\sweepStart, 200,
				\sweepEnd, 3000
			]);
			"→ Played softWhoosh".postln;
		}

		{ soundType == \bird } {
			Synth(\morningBird, [
				\amp, amp * 0.6,
				\freqA, note * 0.8,
				\freqB, note * 1.5
			]);
			"→ Played morningBird".postln;
		}

		{ soundType == \owl } {
			Synth(\owlHoot, [
				\amp, amp,
				\freq, note * 0.5,  // Lower for owl
				\tremoloRate, 6
			]);
			"→ Played owlHoot".postln;
		}

		// ========== FALLBACK ==========
		{ true } {
			"⚠ Unknown instrument: %".format(soundType).postln;
			"Available: harp, flute, ocarina, pad, sparkle, chime,".postln;
			"           brook, rain, fire, leaves, whoosh, bird, owl".postln;
		};

		"============================".postln;

	}, '/play_sound');

	"".postln;
	"✓ OSC Listener '/play_sound' is active on port 57120".postln;
	"✓ Listening for ALL 13 synth types:".postln;
	"".postln;
	"  MELODIC (4): harp, flute, ocarina, pad".postln;
	"  MAGIC (2):   sparkle, chime".postln;
	"  NATURE (4):  brook, rain, fire, leaves".postln;
	"  SFX (3):     whoosh, bird, owl".postln;
	"".postln;
	"Waiting for messages from C++ application...".postln;
	"".postln;
});
)