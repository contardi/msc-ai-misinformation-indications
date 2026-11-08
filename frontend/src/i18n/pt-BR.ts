function pct(n: number): string {
  return `${n.toFixed(1).replace('.', ',')}%`
}

export const ptBR = {
  meta: {
    code: 'pt-BR',
    label: 'Português',
    flag: '🇧🇷',
    htmlLang: 'pt-BR',
    documentTitle: 'Já checaram isso? · Verificação de mensagens',
  },

  header: {
    eyebrow: 'Ferramenta experimental · PT-BR',
    title: 'Já checaram isso?',
    lede:
      'Recebeu uma mensagem, um áudio ou um vídeo e ficou na dúvida? Veja se agências de checagem brasileiras já publicaram alguma coisa sobre o assunto.',
  },

  artefact: {
    eyebrow: 'Trabalho acadêmico',
    body:
      'Este site é o artefato de uma dissertação de mestrado, não um serviço público de checagem. Ele não determina se um conteúdo é verdadeiro ou falso. Na avaliação, cerca de um terço das notícias políticas legítimas foi sinalizado por engano. Use o resultado como ponto de partida para investigar, nunca como conclusão.',
  },

  input: {
    label: 'Cole a mensagem, ou solte aqui o áudio ou vídeo',
    placeholder: 'Ex.: Recebi isso num grupo e queria saber se já checaram…',
    chooseFile: 'Escolher arquivo',
    remove: 'Remover',
    privacy:
      'Nada do que você enviar é armazenado. O áudio é transcrito neste servidor e apagado em seguida.',
    submit: 'Verificar',
    submitting: 'Verificando…',
    tooLarge: (mb: string, limit: number) =>
      `Arquivo muito grande: ${mb} MB. O limite é ${limit} MB.`,
    workingMedia: 'Transcrevendo o áudio…',
    workingText: 'Procurando checagens…',
    mediaSlow: 'Pode levar cerca de um minuto para cada cinco minutos de gravação.',
  },

  transcript: {
    region: 'Transcrição',
    eyebrow: 'O que ouvimos',
    audio: 'áudio',
    video: 'vídeo',
    caveat: 'A busca abaixo usou este texto. Se a transcrição estiver errada, o resultado também estará.',
  },

  cascade: {
    region: 'O que encontramos',
    eyebrow: 'O que encontramos',
    nextStepsLabel: 'O que você pode fazer',
    tiers: {
      evidence: {
        headline: 'Encontramos checagens já publicadas sobre isso.',
        detail:
          'Esta é a resposta mais forte que esta ferramenta pode dar: a conclusão de uma equipe de jornalistas, com autor e link. Leia a checagem antes de tirar suas próprias conclusões.',
      },
      indicators: {
        headline: 'Nenhuma checagem publicada foi encontrada, mas notamos indícios no texto.',
        detail:
          'Os indícios abaixo dizem respeito a COMO o texto foi escrito, e não se ele é verdadeiro. Um texto verdadeiro pode ser escrito de forma sensacionalista, e um texto falso pode ser escrito com sobriedade.',
        nextSteps: [
          'Procure a notícia no site de um veículo que você já conhece.',
          'Veja se outros veículos publicaram a mesma informação.',
          'Desconfie de mensagens que pedem para você compartilhar com urgência.',
        ],
      },
      classifier: {
        headline: 'Nenhuma checagem publicada e nenhum indício de linguagem sensacionalista.',
        detail:
          'Resta a estimativa do modelo, mostrada abaixo, que é a evidência mais fraca desta página. Em textos curtos ela erra com frequência, tanto deixando passar quanto acusando conteúdo legítimo. Não trate esse número como um veredito.',
        nextSteps: [
          'Procure a notícia no site de um veículo que você já conhece.',
          'Veja se outros veículos publicaram a mesma informação.',
          'Verifique a data: conteúdo antigo costuma voltar a circular fora de contexto.',
          'Se puder, procure a fonte original citada no texto.',
        ],
      },
      none: {
        headline: 'Nenhuma checagem publicada foi encontrada sobre este conteúdo.',
        detail:
          'Isso não significa que o conteúdo seja verdadeiro nem que seja falso. Significa apenas que nenhuma das agências consultadas publicou uma checagem sobre ele.',
        nextSteps: [
          'Procure a notícia no site de um veículo que você já conhece.',
          'Veja se outros veículos publicaram a mesma informação.',
          'Verifique a data: conteúdo antigo costuma voltar a circular fora de contexto.',
          'Se puder, procure a fonte original citada no texto.',
        ],
      },
    },
  },

  evidence: {
    eyebrow: 'Checagens publicadas',
    none: 'Nenhuma encontrada',
    noneTitle: 'Não encontramos checagem publicada sobre isso.',
    noneEmphasis: 'Isso não quer dizer que seja verdade.',
    noneDetail:
      'Pode ser um conteúdo novo, pouco divulgado, ou que os verificadores ainda não analisaram. A busca cobre os verificadores brasileiros listados abaixo e não é exaustiva.',
    foundTitle: 'Agências de checagem brasileiras já publicaram sobre alegações parecidas.',
    read: 'Ler a checagem',
    unrated: 'sem classificação',
    attribution:
      'A classificação acima é do verificador, não desta ferramenta. Leia a checagem para o contexto completo.',
  },

  score: {
    region: 'Avaliação',
    eyebrow: 'Avaliação',
    gaugeLabel: 'Probabilidade estimada de desinformação',
    uncertainty: 'Avaliação probabilística, não um veredito.',
    gaugeLow: 'menor probabilidade',
    gaugeHigh: 'maior probabilidade',
    bands: {
      low: {
        chip: 'Poucos sinais de desinformação',
        sentence: (pct: number) =>
          `Estimamos ${pct}% de probabilidade de desinformação: sinais baixos neste texto.`,
      },
      inconclusive: {
        chip: 'Inconclusivo',
        sentence: (pct: number) =>
          `Resultado inconclusivo (${pct}%): não há confiança suficiente para indicar desinformação.`,
      },
      high: {
        chip: 'Sinais associados a desinformação',
        sentence: (pct: number) =>
          `Estimamos ${pct}% de probabilidade de este texto conter desinformação.`,
      },
    },
  },

  corroboration: {
    region: 'Contexto',
    eyebrow: 'Contexto',
    disabled: 'Verificação em verificadores externos não está ativada nesta instância.',
    foundIn: 'Há checagem relacionada em:',
    notFoundIn: 'Não encontramos checagem relacionada nos seguintes verificadores:',
    caveat:
      'Apenas contexto. A ausência de checagem não é prova de falsidade nem entra na pontuação (o índice consultado não é exaustivo).',
  },

  explanationPanel: {
    region: 'Explicação',
    eyebrow: 'Explicação',
    body: (pct: number, terms: string[]) =>
      `O modelo estima ${pct}% de probabilidade de o texto conter desinformação. Trata-se de uma ` +
      `estimativa probabilística, não de um veredito sobre a veracidade do conteúdo.` +
      (terms.length
        ? ` Os termos que mais influenciaram esta estimativa foram: ${terms.join(', ')}.`
        : ''),
    topTokens: 'Termos mais influentes',
    shapNote:
      'Valores SHAP: a contribuição de cada termo para a estimativa do modelo. Termos influentes não são prova.',
  },

  indicators: {
    region: 'Pistas de estilo',
    eyebrow: 'Pistas de estilo',
    cues: {
      exclamations: {
        label: 'Pontuação sensacionalista',
        detail: (n: number) => `${n} pontos de exclamação.`,
      },
      sensational_terms: {
        label: 'Termos de apelo',
        detail: (terms: string[]) => `Termos como: ${terms.join(', ')}.`,
      },
    },
    caveat:
      'Padrões de escrita que aparecem mais em desinformação nos corpora avaliados. São apenas contexto, não provas, e não entram na pontuação.',
    evidence: (fakeRate: number, realRate: number, corpus: string) =>
      `Aparece em ${pct(fakeRate)} dos textos falsos e ${pct(realRate)} dos verdadeiros no corpus ${corpus}.`,
    none: {
      title: 'Nenhum indício de linguagem sensacionalista encontrado.',
      limit: (oneIn: number) =>
        `Esta verificação encontra indícios em cerca de 1 a cada ${oneIn} textos falsos, portanto a ausência de indícios não significa que o conteúdo seja confiável.`,
      next: 'Verifique as checagens publicadas acima e procure a fonte original antes de compartilhar.',
    },
  },

  history: {
    region: 'Histórico',
    eyebrow: 'Histórico · neste navegador',
    clear: 'Limpar histórico',
    replay: 'Rever este resultado',
    note:
      'Armazenado apenas neste navegador (localStorage). Nada é enviado nem armazenado em um servidor; rever um item não reenvia o texto.',
  },

  feedback: {
    region: 'Contestar',
    eyebrow: 'Contestar',
    question: 'Este resultado está errado?',
    reasons: {
      evidence_irrelevant: 'As checagens não têm relação com o texto',
      evidence_missing: 'Faltou uma checagem que eu conheço',
      score_too_high: 'A estimativa parece alta demais',
      score_too_low: 'A estimativa parece baixa demais',
      explanation_unclear: 'Não entendi a explicação',
      other: 'Outro motivo',
    },
    submit: 'Enviar',
    privacy:
      'Enviamos apenas a opção escolhida. O texto analisado não é enviado nem armazenado, e não registramos quem enviou.',
    thanks: 'Obrigado. Sua resposta foi registrada.',
    failed: 'Não foi possível registrar sua resposta agora.',
  },

  footer: {
    purpose:
      'Esta ferramenta procura checagens já publicadas. Ela não decide o que é verdade e não substitui a leitura da checagem original.',
    academic:
      'Artefato acadêmico (Mestrado em Inteligência Artificial). O conteúdo enviado não é armazenado pelo servidor; o histórico fica apenas no seu navegador.',
  },

  errors: {
    unexpected: 'Erro inesperado.',
    network: 'Não foi possível contatar o serviço. Verifique sua conexão.',
    mediaNetwork:
      'Não foi possível concluir o envio. Arquivos muito longos ou grandes podem não terminar em tempo. Tente um trecho mais curto.',
    modelMissing: 'O modelo ainda não está disponível. É necessário treinar o modelo (ver README).',
    failed: (status: number) => `A análise falhou (HTTP ${status}).`,
    rateLimited: 'Muitos envios em pouco tempo. Aguarde um momento e tente novamente.',
    mediaFailed: 'Não foi possível processar este arquivo.',
    feedbackFailed: 'Não foi possível registrar sua resposta agora.',
  },

  language: { switchTo: 'Mudar idioma', current: 'Idioma' },
}

export type Dictionary = typeof ptBR
