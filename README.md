# NAACL_BioNLP_2021_MEDIQA_QS_NCU
This study describes the model design of the NCUEE-NLP system for the MEDIQA challenge at the BioNLP 2021 workshop. We use the PEGASUS transformers and fine-tune the downstream summarizatio task using our collected and processed datasets.  A total of 22 teams participated in the consumer health question summarization task of MEDIQA 2021. Each participating team was allowed to submit a maximum of ten runs. Our best submission, achieving a ROUGE2-F1 score of 0.1597, ranked third among all 128
submissions.

- Website: https://sites.google.com/view/mediqa2021
- Our paper: https://aclanthology.org/2021.bionlp-1.30.pdf
- Overview Paper: https://www.aclweb.org/anthology/2021.bionlp-1.8.pdf
- Media News Report: http://ncusec.ncu.edu.tw/news/event_content.php?E_ID=285

## Tasks
Task 1 (question summarization): https://github.com/abachaa/MEDIQA2021/tree/main/Task1

## Evaluation
- The AIcrowd platform was used for releasing the test sets and submitting runs.
- ROUGE was used as the main metric to rank the participating teams, several evaluation metrics such as HOLMS and CheXbert were also used to be more adapted to each task.
