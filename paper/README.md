# JustFit v2 paper

**JustFit v2: Long-Context Local LLM Serving with Just-in-Time State Management**

Yuhua Chen · Preprint · October 31, 2026 · 32 pages

- [Read the paper (PDF)](https://github.com/YuhuaBillChen/justfit-llm/blob/main/paper/JustFit-v2.pdf)
- arXiv: submitted; the link will be added here when it is public
- [Download clean LaTeX source](JustFit-v2-arxiv-source.zip)
- [BibTeX](citation.bib) · [CITATION.cff](../CITATION.cff)
- Contact: [contact@yuhuachen.com](mailto:contact@yuhuachen.com)

This revision reorganizes the paper into background, system design, setup and evaluation, and adds the recorded coding-agent case (Token Garden) and acknowledgments. It labels the historical MLX-VLM baseline (TQ4 KV and MTP) and its completed 30K image service workload, reports the Splash 1.3.0 72K figure as its memory planner's maximum, and credits TurboQuant where TQ4 is introduced. Measurements, task scores and the AIME 2026 comparison (95.6% / 94.4% over 30 questions at three seeds) are unchanged.

This directory contains only the current paper and its LaTeX source. Updates replace these files at the same paths; previous revisions remain in Git history.

The source archive contains LaTeX, the generated bibliography, template files and the 19 referenced vector figures. It builds with pdfLaTeX using the included `paper.bbl`; no CSV data or internal experiment receipts are included. It is the source prepared for the arXiv submission.

The software's Apache-2.0 license is not a grant of that license to this manuscript. Copyright © 2026 Yuhua Chen; manuscript reuse rights remain with the author pending a separate paper license choice. Third-party template files retain their own terms.

If this work supports your research, please cite:

```bibtex
@unpublished{chen2026justfitv2,
  author = {Chen, Yuhua},
  title = {{JustFit v2}: Long-Context Local LLM Serving with Just-in-Time State Management},
  year = {2026},
  month = oct,
  note = {Preprint},
  url = {https://github.com/YuhuaBillChen/justfit-llm/blob/main/paper/JustFit-v2.pdf}
}
```
