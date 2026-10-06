# protransformers.github.io

Source of the ProTransformer website: **https://protransformers.github.io**

- Code: https://github.com/chris-hzc/ProTransformer
- Package: `pip install protransformers`
- Paper: [ProTransformer: Robustify Transformers via Plug-and-Play Paradigm](https://arxiv.org/abs/2410.23182) (NeurIPS 2024)

Static site (HTML/CSS/JS, no build step). Preview locally with `python -m http.server` and open http://localhost:8000.
`data/text_examples.json` holds real TextFooler examples crafted against `textattack/bert-base-uncased-ag-news`, scored by BERT and Pro-BERT (MCP, γ=4, K=3) with the same weights.
