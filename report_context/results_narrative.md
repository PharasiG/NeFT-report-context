### 4.1 Per-Experiment Results Summary

The final result tables covered 16 POS and NER experiment keys across mBERT and XLM-R. One experiment, `XLM_french_pos`, contained only the baseline row, so direct baseline-versus-masked comparison was available for 15 experiments. Within those comparable experiments, the best masked configuration exceeded the baseline accuracy in every case. The improvement was modest in absolute terms, but it was consistent across both task families.

The POS experiments were numerically stable across percentages. Baseline accuracies already ranged from 0.9617 to 0.9850, and the best overall accuracies rose to 0.9640 for `mBERT_Chinese_pos`, 0.9754 for `mBERT_hindi_pos`, 0.9690 for `mBERT_arabic_pos`, 0.9838 for `mBERT_french_pos`, 0.9807 for `XLM_hindi_pos`, 0.9730 for `XLM_arabic_pos`, and 0.9698 for `XLM_chinese_pos`. The associated winning deltas over baseline lay between 0.0011 and 0.0028, indicating that POS performance changed gradually rather than abruptly across the percentage sweep.

The NER experiments showed larger gains and greater variability. Best overall accuracies reached 0.9582 for `mBERT_hindi_ner`, 0.9811 for `mBERT_arabic_ner`, 0.9353 for `mBERT_chinese_ner`, 0.9543 for `mBERT_french_ner`, 0.9610 for `XLM_hindi_ner`, 0.9828 for `XLM_arabic_ner`, 0.9288 for `XLM_chinese_ner`, and 0.9512 for `XLM_french_ner`. The winning deltas over baseline ranged from 0.0035 to 0.0059, which was materially larger than the POS range. The raw tables also showed that some NER settings were more sensitive to the selected percentage than POS, particularly in Arabic, where macro F1 varied substantially across percentages while accuracy remained within a narrow band.

### 4.2 Cross-Method Comparison

Across the experiment-wise winners in Table 2, Probeless achieved the best overall accuracy in 9 experiments, NeFT in 6 experiments, and baseline in 1 experiment. The baseline-only outcome occurred in `XLM_french_pos`, where no masked rows were available. On the remaining comparable experiments, the aggregate method summary in Table 4c showed that Probeless had the highest average best accuracy (0.9643) and the highest average best macro F1 (0.8882), followed by NeFT at 0.9639 and 0.8861, and then Baseline at 0.9626 and 0.8835. Thus, at the summary level, both masked methods outperformed the baseline, and Probeless held a small aggregate advantage over NeFT.

The method comparison was task-dependent. In NER, Probeless was the stronger method in 6 of 8 experiments, while NeFT was best in 2 of 8. In POS, the comparison was more balanced: Probeless was best in 3 experiments, NeFT in 3, one experiment was effectively tied in best accuracy (`XLM_arabic_pos`, recorded under NeFT in the summary), and one experiment had baseline only. This pattern indicated that the relative advantage of Probeless was clearer in NER than in POS.

The improvement-over-baseline table reinforced the same pattern. In POS, the better of NeFT or Probeless exceeded baseline by 0.0011 to 0.0028. In NER, the corresponding gains were 0.0035 to 0.0059. The numerical separation between methods was usually small, but the overall comparison favored masked training over baseline and favored Probeless slightly over NeFT.

### 4.3 Best-Performing Configurations

The best-performing percentage was not constant across experiments. For NeFT, the best percentage ranged from 5 to 50 depending on the experiment. Examples included 50 for `XLM_hindi_pos`, 35 for `mBERT_Chinese_pos`, 15 for `XLM_french_ner`, 40 for `mBERT_chinese_ner`, and 45 for `XLM_hindi_ner`. For Probeless, the best percentage also varied widely, including 15 for `mBERT_french_pos`, 20 for `mBERT_arabic_ner`, 35 for `XLM_arabic_ner`, 40 for `XLM_chinese_ner`, and 45 for `XLM_french_ner`. No single percentage emerged as uniformly optimal across tasks, models, or languages.

At the cross-scope level, Table 4a showed that the average best accuracy was higher for POS (0.9751) than for NER (0.9566), and the average best macro F1 was also higher for POS (0.9490) than for NER (0.8345). These averages should be interpreted as task-level summaries rather than direct indicators of task difficulty, but they show that the reported POS scores occupied a higher numerical range overall.

Table 4b showed a slight aggregate advantage for XLM-R over mBERT when the best result per experiment was considered. XLM-R had an average best accuracy of 0.9665 and an average best macro F1 of 0.8967, compared with 0.9651 and 0.8868 for mBERT. The margin was not large, but it was directionally consistent across both summary metrics.

### 4.4 Key Quantitative Observations

First, the results supported a consistent overall advantage of masked fine-tuning over the baseline wherever masked comparisons were available. In 15 comparable experiments, the best masked configuration always matched or exceeded baseline accuracy, and in most cases the best result came from either NeFT or Probeless rather than the baseline.

Second, the magnitude of gain depended on the task family. POS improvements were consistently positive but small, while NER gains were larger. This pattern was visible both in the experiment-wise deltas and in the broader spread of NER raw results across percentages.

Third, the choice of neuron percentage materially affected performance, but the optimal percentage was experiment-specific rather than global. High-performing settings appeared at low, mid, and high percentages across different experiments, which indicates that the percentage effect was not monotonic in the final tables.

Fourth, cross-language stability differed by task. The POS experiments were comparatively stable across languages and percentages, with narrow accuracy bands and relatively small method differences. The NER experiments were less uniform, and Arabic NER in particular showed notable variation in macro F1 across settings despite limited movement in accuracy. Chinese, French, and Hindi NER were more stable than Arabic in this respect.

Finally, the cross-model comparison modestly favored XLM-R in the aggregated best-result summaries, but both backbones benefited from masked fine-tuning. The stronger distinction in the final tables was therefore not between mBERT and XLM-R, but between baseline versus masked training, and between the more stable POS setting and the more variable NER setting.
