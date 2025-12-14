// Scholarly Project: Latent Sound Atlas
// Bhavya Venkataraghavan

(
s.waitForBoot({
	var moods, synthDistribution, synthMap, categoryMap, scaleMap;
    var logFile, safeLog, renderQueue, totalFileCount;
    var rootDir;
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
	// CONFIGURATION
	// =========================================================================

	moods = #[\cozy, \mystic, \tense];

	synthDistribution = (
		harp: (total: 18, perMood: 6, singleCount: 6),
	 	flute: (total: 18, perMood: 6, singleCount: 6),
	 	ocarina: (total: 27, perMood: 9, singleCount: 3, melodyCount: 6),
	 	pad: (total: 15, perMood: 5, singleCount: 0, melodyCount: 5),
	 	sparkle: (total: 15, perMood: 5, singleCount: 5),
	 	chime: (total: 15, perMood: 5, singleCount: 5),
	 	brook: (total: 6, perMood: [2, 2, 2]),
	 	rain: (total: 6, perMood: [2, 2, 2]),
	 	fire: (total: 6, perMood: [2, 2, 2]),
	 	leaves: (total: 6, perMood: [2, 2, 2]),
	 	whoosh: (total: 6, perMood: [2, 2, 2]),
	 	bird: (total: 6, perMood: [2, 2, 2]),
	 	owl: (total: 6, perMood: [2, 2, 2])
	);

	synthMap = (
	 	harp: \celticHarp, flute: \forestFlute, ocarina: \ocarinaNote, pad: \mysticPad,
	 	sparkle: \fairySparkle, chime: \fairyChime,
	 	brook: \babblingBrook, rain: \relaxingRain, fire: \fireCrackle, leaves: \rustlingLeaves,
	 	whoosh: \softWhoosh, bird: \morningBird, owl: \owlHoot
	);

	scaleMap = (
		\cozy: Scale.majorPentatonic.degrees,
		\mystic: Scale.dorian.degrees,
		\tense: Scale.minor.degrees
	);

	categoryMap = (
		harp: \mus, flute: \mus, ocarina: \mus, pad: \mus,
		sparkle: \fx, chime: \fx,
		brook: \amb, rain: \amb, fire: \amb, leaves: \amb,
		whoosh: \sfx, bird: \sfx, owl: \sfx
	);

	// =========================================================================
	// OUTPUT SETUP
	// =========================================================================

	rootDir = PathName(thisProcess.nowExecutingPath).parentPath;
	~outputFolder = PathName(rootDir ++ "sound_assets").fullPath;
	File.mkdir(~outputFolder);

	logFile = File(rootDir +/+ "render_log.txt", "a");
	safeLog = { |msg|
		{ logFile.write(msg ++ "\n"); logFile.flush; }.try({ "LOGGING FAILED".postln });
	};

	// =========================================================================
	// BUILD QUEUE
	// =========================================================================

	renderQueue = List.new;
	totalFileCount = 0;

	synthDistribution.keysValuesDo { |synthKey, data|
		var perMood = data[\perMood];
		var singleCount = data[\singleCount];
		var melodyCount = data[\melodyCount];
		var moodFileIndex = 0;

		moods.do { |mood, moodIndex|
	 		var numFiles = if(perMood.isKindOf(Array)) {
	 			perMood[moodIndex]
	 		} {
	 			perMood
	 		};

	 		if (singleCount.notNil) {
	 			singleCount.do { |i|
	 				renderQueue.add((
	 					synth: synthKey, mood: mood, type: \single,
	 					index: moodFileIndex, totalIndex: totalFileCount
	 				));
	 				moodFileIndex = moodFileIndex + 1;
	 				totalFileCount = totalFileCount + 1;
	 			};

	 			melodyCount.do { |i|
	 				renderQueue.add((
	 					synth: synthKey, mood: mood,
	 					type: if (synthKey == \pad) { \texture } { \melody },
	 					index: moodFileIndex, totalIndex: totalFileCount
	 				));
	 				moodFileIndex = moodFileIndex + 1;
	 				totalFileCount = totalFileCount + 1;
	 			};
	 		} {
	 			numFiles.do { |i|
	 				renderQueue.add((
	 					synth: synthKey, mood: mood, type: \texture,
	 					index: moodFileIndex, totalIndex: totalFileCount
	 				));
	 				moodFileIndex = moodFileIndex + 1;
	 				totalFileCount = totalFileCount + 1;
	 			};
	 		};
		};
	};

	"Queue built: % tasks.".format(totalFileCount).postln;

	// =========================================================================
	// RENDER LOOP
	// =========================================================================

	fork {
	 	var filesProduced = 0;

	 	"*** BEGINNING BATCH RENDER: % files total ***".format(totalFileCount).postln;
	 	safeLog.("=== BATCH RENDER START: % files ===".format(totalFileCount));

	 	renderQueue.do { |task, i|
	 		var synthName = synthMap[task[\synth]];
			var moodScale = scaleMap[task[\mood]];
	 		var synthKey = task[\synth];
	 		var mood = task[\mood];
	 		var index = task[\index];
	 		var totalIndex = task[\totalIndex];
	 		var typeStr, typeStrPost, prefix, fileName, fullPath;
	 		var score, duration, note, scoreEvents, isTense, finalStop;

	 		// Filename
	 		typeStr = switch(task[\type],
	 			\single, { "single" },
	 			\melody, { "melody" },
	 			\texture, { "texture" },
	 			{ "" }
	 		);
			typeStrPost = if(typeStr.notEmpty) { typeStr ++ "_" } { "" };
	 		prefix = categoryMap[synthKey].asString ++ "_";
	 		fileName = "%%%%_%.wav".format(
	 			prefix ++ synthKey.asString, typeStrPost,
	 			mood.asString, index.asString.padLeft(3, "0")
	 		);
	 		fullPath = ~outputFolder +/+ fileName;

			safeLog.("BEGIN: % / % / % (%/% total)".format(mood, synthKey, index, totalIndex + 1, totalFileCount));

	 		// Generate Score
	 		duration = rrand(5.0, 10.0);
	 		scoreEvents = List.new;
	 		isTense = mood == \tense;

	 		case
	 		{ synthName == \celticHarp && task[\type] == \single } {
	 			note = (moodScale.choose + 65).midicps;
	 			duration = 6.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, note, \dur, duration - 0.5, \amp, rrand(3.0, 5.0),
	 			\decayArg, rrand(4.0, 7.0), \verbMix, rrand(0.6, 0.9)]]);
	 			scoreEvents.add([duration + 0.1, [\c_set, 0, 0]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \forestFlute && task[\type] == \single } {
	 			note = (moodScale.choose + 60).midicps;
	 			duration = 4.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, note, \dur, 2.0, \amp, rrand(0.3, 0.6), // Safe Amp
	 			\breathLevel, rrand(0.1, 0.3), \cutoff, rrand(3500, 5000)]]);
	 			scoreEvents.add([duration + 0.1, [\c_set, 0, 0]]);
	 			finalStop = 2.5;
	 		}
	 		{ synthName == \ocarinaNote && task[\type] == \single } {
	 			note = (moodScale.choose + 63).midicps;
	 			duration = 4.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, note, \dur, 2.0, \amp, rrand(0.3, 0.6),
	 			\vibRate, rrand(3.0, 6.0), \vibDepth, rrand(0.003, 0.007)]]);
	 			scoreEvents.add([duration + 0.1, [\c_set, 0, 0]]);
	 			finalStop = 2.5;
	 		}
	 		{ (synthName == \celticHarp || synthName == \forestFlute || synthName == \ocarinaNote) && task[\type] == \melody } {
	 			var numNotes = rrand(2, 5);
	 			var noteDur = duration / (numNotes * 1.5);
	 			var melodyTime = 0.0;
	 			var lastNoteEndTime = 0.0;

	 			numNotes.do { |n|
	 				note = (moodScale.choose + 60 + [0, 12].choose).midicps;

	 				case
	 				{ synthName == \celticHarp } {
	 					scoreEvents.add([melodyTime, [\s_new, synthName, 1000 + n, 0, 0,
	 						\freq, note, \dur, noteDur * 1.5,
	 						\amp, rrand(0.3, 0.6),
	 					\decayArg, rrand(4.0, 7.0), \verbMix, rrand(0.6, 0.9)]]);
	 					lastNoteEndTime = melodyTime + (noteDur * 1.5);
	 				}
	 				{ synthName == \forestFlute } {
	 					scoreEvents.add([melodyTime, [\s_new, synthName, 1000 + n, 0, 0,
	 						\freq, note, \dur, noteDur,
	 						\amp, rrand(0.3, 0.6),
	 					\breathLevel, rrand(0.1, 0.3), \cutoff, rrand(3500, 5000)]]);
	 					lastNoteEndTime = melodyTime + noteDur;
	 				}
	 				{ synthName == \ocarinaNote } {
	 					scoreEvents.add([melodyTime, [\s_new, synthName, 1000 + n, 0, 0,
	 						\freq, note, \dur, noteDur,
	 						\amp, rrand(0.3, 0.6),
	 					\vibRate, rrand(3.0, 6.0), \vibDepth, rrand(0.003, 0.007)]]);
	 					lastNoteEndTime = melodyTime + noteDur;
	 				};

	 				melodyTime = melodyTime + (noteDur * 1.2);
	 			};

	 			finalStop = lastNoteEndTime + 1.0;
	 		}
	 		{ synthName == \mysticPad } {
	 			note = (moodScale.choose + 40).midicps;
	 			duration = 10.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, note, \dur, duration, \amp, rrand(0.3, 0.6),
	 				\cutoffBase, isTense.if(200, rrand(400, 600)),
	 			\cutoffRange, isTense.if(300, rrand(700, 1200))]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \fairySparkle } {
	 			duration = rrand(5.0, 8.0);
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, rrand(1500, 3500), \dur, duration, \amp, rrand(0.3, 0.6),
	 			\density, rrand(1, 20)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \fairyChime } {
	 			duration = rrand(5.0, 8.0);
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\freq, rrand(1800, 3000), \dur, duration, \amp, rrand(0.3, 0.6),
	 			\density, rrand(15, 30), \verbMix, rrand(0.4, 0.7)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \babblingBrook } {
	 			duration = 10.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\dur, duration, \amp, rrand(4.0, 8.0), // Safe Amp
	 			\filterFreq, rrand(10.0, 20.0), \bubbleGain, rrand(0.1, 0.3)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \relaxingRain } {
	 			duration = 10.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\dur, duration, \amp, rrand(0.2, 0.4),
	 			\brownMix, rrand(0.2, 0.6), \cutoffLow, rrand(200, 450)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \fireCrackle } {
	 			var density = rrand(5.0, 15.0);
	 			var lowPass = density.linlin(5, 15, 400, 100);
	 			duration = 8.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\dur, duration, \amp, rrand(0.2, 0.4),
	 			\density, density, \lowPass, lowPass]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \rustlingLeaves } {
	 			duration = 8.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\dur, duration, \amp, rrand(0.3, 0.5),
	 			\modRate, rrand(0.5, 2.0), \filterCenter, rrand(1000, 2500)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \softWhoosh } {
	 			duration = rrand(1.5, 3.0);
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\dur, duration, \amp, rrand(0.4, 0.7),
	 			\sweepStart, rrand(100, 500), \sweepEnd, rrand(2500, 4000)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \morningBird } {
	 			duration = 3.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\pan, [-0.8, 0.8].choose, \amp, rrand(0.1, 0.3),
	 			\freqA, rrand(1000, 2250), \freqB, rrand(2500, 4500)]]);
	 			finalStop = duration + 0.5;
	 		}
	 		{ synthName == \owlHoot } {
	 			duration = 2.0;
	 			scoreEvents.add([0.0, [\s_new, synthName, 1000, 0, 0,
	 				\pan, [-0.8, 0.8].choose, \amp, rrand(0.1, 0.3),
	 			\freq, rrand(400, 650), \tremoloRate, rrand(4.0, 8.0)]]);
	 			finalStop = duration + 0.5;
			};

	 		// --- General Cleanup Section ---
	 		scoreEvents.add([finalStop, [\n_free, -1]]); // Free ALL Synths
	 		scoreEvents.add([finalStop + 0.01, [\quit]]); // Stop the server and finalize the file
	 		scoreEvents.add([finalStop + 0.02, [\c_set, 0, 0]]);

	 		// Render
	 		score = Score(scoreEvents);
	 		score.recordNRT(
	 			outputFilePath: fullPath,
	 			sampleRate: 44100,
	 			headerFormat: "WAV",
	 			sampleFormat: "int16",
	 			options: ServerOptions.new.numOutputBusChannels_(2),
	 			action: {
	 				safeLog.("DONE: % / % / % (%/% total)".format(mood, synthKey, index, totalIndex + 1, totalFileCount));
	 				("Rendered: " ++ fileName ++ " (" ++ (totalIndex + 1) ++ "/150)").postln;
	 			}
	 		);

	 		0.02.wait;
	 		filesProduced = filesProduced + 1;
	 	};

	 	logFile.close;
	 	"*** BATCH RENDERING COMPLETE! % files successfully created. ***".format(filesProduced).postln;
	 	"Output folder: %".format(~outputFolder).postln;
	 	safeLog.("=== BATCH RENDER COMPLETE: % files ===".format(filesProduced));
	};
});
)