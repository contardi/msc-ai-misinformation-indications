import type { Dictionary } from './pt-BR'

function pct(n: number): string {
  return `${n.toFixed(1)}%`
}

export const enGB: Dictionary = {
  meta: {
    code: 'en-GB',
    label: 'English',
    flag: '🇬🇧',
    htmlLang: 'en-GB',
    documentTitle: 'Has this been checked? · Brazilian fact-check search',
  },

  header: {
    eyebrow: 'Experimental tool · Brazilian Portuguese content',
    title: 'Has this been checked?',
    lede:
      'Received a message, a voice note or a video and unsure about it? See whether Brazilian fact-checking agencies have already published something on the subject. Works on Portuguese-language content.',
  },

  artefact: {
    eyebrow: 'Academic work',
    body:
      'This site is an MSc dissertation artefact, not a public fact-checking service. It does not determine whether content is true or false. In evaluation, about a third of legitimate political articles were flagged in error. Treat the result as a starting point for your own checking, never as a conclusion.',
  },

  input: {
    label: 'Paste the message, or drop an audio or video file here',
    placeholder: 'e.g. Someone forwarded this to a group and I want to know if it has been checked…',
    chooseFile: 'Choose a file',
    remove: 'Remove',
    privacy:
      'Nothing you send is stored. Audio is transcribed on this server and deleted immediately afterwards.',
    submit: 'Check it',
    submitting: 'Checking…',
    tooLarge: (mb: string, limit: number) => `File too large: ${mb} MB. The limit is ${limit} MB.`,
    workingMedia: 'Transcribing the audio…',
    workingText: 'Searching for fact-checks…',
    mediaSlow: 'This takes roughly a minute for every five minutes of recording.',
  },

  transcript: {
    region: 'Transcript',
    eyebrow: 'What we heard',
    audio: 'audio',
    video: 'video',
    caveat:
      'The search below used this text. If the transcription is wrong, so is the result.',
  },

  cascade: {
    region: 'What we found',
    eyebrow: 'What we found',
    nextStepsLabel: 'What you can do',
    tiers: {
      evidence: {
        headline: 'We found fact-checks already published about this.',
        detail:
          'This is the strongest answer this tool can give: the conclusion of a team of journalists, with an author and a link. Read the check before drawing your own conclusions.',
      },
      indicators: {
        headline: 'No published fact-check was found, but we noticed signals in the text.',
        detail:
          'The signals below are about HOW the text is written, not whether it is true. A true story can be written sensationally, and a false one can be written soberly.',
        nextSteps: [
          'Look for the story on the site of an outlet you already know.',
          'Check whether other outlets published the same information.',
          'Be wary of messages urging you to share them urgently.',
        ],
      },
      classifier: {
        headline: 'No published fact-check, and no sensationalist wording signals.',
        detail:
          "What remains is the model's estimate, shown below, which is the weakest evidence on this page. On short text it is wrong often, both letting things through and accusing legitimate content. Do not treat that number as a verdict.",
        nextSteps: [
          'Look for the story on the site of an outlet you already know.',
          'Check whether other outlets published the same information.',
          'Check the date: old content often returns to circulation out of context.',
          'If you can, find the original source cited in the text.',
        ],
      },
      none: {
        headline: 'No published fact-check was found about this content.',
        detail:
          'That does not mean the content is true, and it does not mean it is false. It means only that none of the agencies consulted has published a check about it.',
        nextSteps: [
          'Look for the story on the site of an outlet you already know.',
          'Check whether other outlets published the same information.',
          'Check the date: old content often returns to circulation out of context.',
          'If you can, find the original source cited in the text.',
        ],
      },
    },
  },

  evidence: {
    eyebrow: 'Published fact-checks',
    none: 'None found',
    noneTitle: 'We found no published fact-check about this.',
    noneEmphasis: 'That does not mean it is true.',
    noneDetail:
      'It may be new, little-circulated, or simply not yet examined. The search covers the Brazilian fact-checkers listed below and is not exhaustive.',
    foundTitle: 'Brazilian fact-checking agencies have already published on similar claims.',
    read: 'Read the fact-check',
    unrated: 'unrated',
    attribution:
      'The rating above is the fact-checker’s, not this tool’s. Read their article for the full context.',
  },

  score: {
    region: 'Assessment',
    eyebrow: 'Assessment',
    gaugeLabel: 'Estimated likelihood of misinformation',
    uncertainty: 'A probabilistic assessment, not a verdict.',
    gaugeLow: 'lower likelihood',
    gaugeHigh: 'higher likelihood',
    bands: {
      low: {
        chip: 'Few signals of misinformation',
        sentence: (pct: number) =>
          `We estimate a ${pct}% likelihood of misinformation: few signals in this text.`,
      },
      inconclusive: {
        chip: 'Inconclusive',
        sentence: (pct: number) =>
          `Inconclusive (${pct}%): not enough confidence to indicate misinformation.`,
      },
      high: {
        chip: 'Signals associated with misinformation',
        sentence: (pct: number) =>
          `We estimate a ${pct}% likelihood that this text contains misinformation.`,
      },
    },
  },

  corroboration: {
    region: 'Context',
    eyebrow: 'Context',
    disabled: 'Checking against external fact-checkers is not enabled on this instance.',
    foundIn: 'A related fact-check exists at:',
    notFoundIn: 'We found no related fact-check at these organisations:',
    caveat:
      'Context only. The absence of a fact-check is not proof of falsehood and does not enter the score (the index consulted is not exhaustive).',
  },

  explanationPanel: {
    region: 'Explanation',
    eyebrow: 'Explanation',
    body: (pct: number, terms: string[]) =>
      `The model estimates a ${pct}% likelihood that this text contains misinformation. This is a ` +
      `probabilistic estimate, not a verdict on whether the content is true.` +
      (terms.length ? ` The terms that most influenced it were: ${terms.join(', ')}.` : ''),
    topTokens: 'Most influential terms',
    shapNote:
      'SHAP values: each term’s contribution to the model’s estimate. An influential term is not proof.',
  },

  indicators: {
    region: 'Style cues',
    eyebrow: 'Style cues',
    cues: {
      exclamations: {
        label: 'Sensationalist punctuation',
        detail: (n: number) => `${n} exclamation marks.`,
      },
      sensational_terms: {
        label: 'Urgency wording',
        detail: (terms: string[]) => `Terms such as: ${terms.join(', ')}.`,
      },
    },
    caveat:
      'Wording patterns that were more common in misinformation across the corpora we evaluated. Context only, never proof, and they do not enter the score.',
    evidence: (fakeRate: number, realRate: number, corpus: string) =>
      `Appears in ${pct(fakeRate)} of false texts and ${pct(realRate)} of legitimate ones in the ${corpus} corpus.`,
    none: {
      title: 'No sensationalist wording found.',
      limit: (oneIn: number) =>
        `This check finds cues in roughly 1 in ${oneIn} false texts, so finding none does not mean the content is reliable.`,
      next: 'Check the published fact-checks above and look for the original source before sharing.',
    },
  },

  history: {
    region: 'History',
    eyebrow: 'History · in this browser',
    clear: 'Clear history',
    replay: 'Show this result again',
    note:
      'Stored in this browser only (localStorage). Nothing is sent to or kept on a server; revisiting an item does not resend the text.',
  },

  feedback: {
    region: 'Contest',
    eyebrow: 'Contest',
    question: 'Is this result wrong?',
    reasons: {
      evidence_irrelevant: 'The fact-checks are unrelated to the text',
      evidence_missing: 'A fact-check I know of is missing',
      score_too_high: 'The estimate reads too high',
      score_too_low: 'The estimate reads too low',
      explanation_unclear: 'I did not understand the explanation',
      other: 'Another reason',
    },
    submit: 'Send',
    privacy:
      'We send only the option you choose. The analysed text is neither sent nor stored, and we do not record who sent it.',
    thanks: 'Thank you. Your response has been recorded.',
    failed: 'Your response could not be recorded right now.',
  },

  footer: {
    purpose:
      'This tool looks for fact-checks that have already been published. It does not decide what is true, and it is no substitute for reading the original.',
    academic:
      'Academic artefact (MSc Artificial Intelligence). Submitted content is not stored by the server; the history stays in your browser only.',
  },

  errors: {
    unexpected: 'Something went wrong.',
    network: 'Could not reach the service. Check your connection.',
    mediaNetwork:
      'The upload could not be completed. Very long or large files may not finish in time. Try a shorter excerpt.',
    modelMissing: 'The model is not available yet. It needs to be trained first (see the README).',
    failed: (status: number) => `The check failed (HTTP ${status}).`,
    rateLimited: 'Too many submissions in a short time. Wait a moment and try again.',
    mediaFailed: 'Could not process this file.',
    feedbackFailed: 'Your response could not be recorded right now.',
  },

  language: { switchTo: 'Change language', current: 'Language' },
}
