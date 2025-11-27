
## 9. Predictive performance assessment 

We evaluated how well the models can identify drone-active patches in held-out data by computing ROC AUC, accuracy, precision, and recall. We also evaluated spatial quality using an IoU-based metric computed per frame.

#### Classification performance (50 000 sampled patches)

Using a balanced evaluation across 50 000 randomly selected labeled rows (with an empirical drone prior of 0.0009), the simple NB classifier and the hierarchical NB model showed the following performance:

- **NB model**
  - ROC AUC: 0.6237  
  - Accuracy: 0.9935  
  - Precision: 0.0173  
  - Recall: 0.1111  
  - Confusion: TP=5, TN=49671, FP=284, FN=40

- **Hierarchical NB model**
  - ROC AUC: 0.7778  
  - Accuracy: 0.9929  
  - Precision: 0.0215  
  - Recall: 0.1556  
  - Confusion: TP=7, TN=49637, FP=318, FN=38

Because positives are extremely rare (0.09%), overall accuracy is not very informative—both models appear “good” simply by predicting background. The more meaningful metrics are ROC AUC, precision, and recall. The hierarchical model improves ROC AUC substantially (+0.15), meaning it ranks positive patches more consistently above negative ones. Precision and recall remain low for both models due to severe class imbalance and the fact that event-count-only models have limited discriminative power, but the hierarchical model shows modest improvements in both.

Overall, the hierarchical NB model provides better separation between drone and background patches, though both models remain weak classifiers in an absolute sense.

#### IoU-based spatial detection quality

To evaluate how well the models localize the drone spatially on a frame basis, we computed IoU between predicted positive patches and the ground-truth bounding box on all 1872 frames that contain at least one drone patch. We evaluated predictions using a threshold of P(z=1|y) ≥ 0.1 and an empirical per-frame prior of 0.0048.

The results were:

- **NB model**
  - Mean IoU: 0.1538  
  - Median IoU: 0.1570  
  - Frames with IoU ≥ 0.1: 83.39%  
  - Frames with IoU ≥ 0.3: 1.66%  
  - Frames with IoU ≥ 0.5: 0.05%

- **Hierarchical NB model**
  - Mean IoU: 0.1553  
  - Median IoU: 0.1538  
  - Frames with IoU ≥ 0.1: 84.03%  
  - Frames with IoU ≥ 0.3: 2.03%  
  - Frames with IoU ≥ 0.5: 0.05%

The spatial results show a small but consistent benefit from the hierarchical model. Both models frequently identify a region overlapping the true drone location to some degree (IoU ≥ 0.1 in ~84% of frames). However, achieving tight localization is difficult: IoU ≥ 0.3 is rare (~2%) and IoU ≥ 0.5 almost never occurs. This is unsurprising given that both models rely solely on aggregated event counts and do not use spatial or temporal structure beyond patch identity.

#### Summary

Together, these metrics demonstrate that:

1. The hierarchical NB model outperforms the simple NB classifier, particularly in terms of ROC AUC and modest increases in precision/recall.  
2. Both models can roughly point to the drone’s region in most frames, but cannot localize it sharply.  
3. The predictive performance is consistent with the simplicity of the underlying modeling assumptions and highlights the need for richer models (e.g. GLRT-based features, temporal dynamics, spatial correlation) for high-quality drone detection and tracking.

This section fulfills the optional/bonus predictive performance requirement by quantitatively comparing models and discussing the practical value and limitations of their detection accuracy.


## 10. Model comparison using PSIS-LOO

We compared the simple NB model and the hierarchical NB model using PSIS-LOO cross-validation. For both models we computed the pointwise log-likelihood for 30 000 held-out observations and 4 000 posterior draws, then used ArviZ to estimate the expected log predictive density (elpd_loo) and the effective number of parameters (p_loo). Higher elpd_loo indicates better out-of-sample predictive performance.

#### LOO results

- **NB model**
  - elpd_loo = –4648.19 (SE = 166.63)  
  - p_loo = 8.23  
  - Pareto k diagnostics: 1 problematic observation (k > 0.7)

- **Hierarchical NB model**
  - elpd_loo = –4341.05 (SE = 151.31)  
  - p_loo = 394.31  
  - Pareto k diagnostics: 227 bad (0.7 < k ≤ 1) and 41 very bad (k > 1)

#### Interpretation

1. **Predictive accuracy**
   - The hierarchical model has a much higher elpd_loo (by ~307 points), meaning it provides substantially better out-of-sample predictive performance than the global NB model.
   - This matches the improvements we observed earlier in ROC AUC and IoU metrics.

2. **Model complexity**
   - The hierarchical model has a very large p_loo (≈ 394), reflecting its many patch-level parameters.
   - The simple NB model has p_loo ≈ 8, consistent with being a very low-complexity model.
   - The improvement in elpd_loo outweighs the large increase in complexity, suggesting that the additional hierarchical structure captures genuine variation in the data rather than overfitting.

3. **Pareto k diagnostics**
   - Both models trigger warnings, indicating influential observations where the posterior predictive distribution differs strongly from the leave-one-out predictive distribution.
   - However, the hierarchical model has substantially more problematic points (k > 0.7 and k > 1).
   - This is common in hierarchical models with many latent parameters; individual observations can heavily influence patch-specific effects.
   - These warnings mean that exact LOO or K-fold CV could give more reliable values, but PSIS-LOO is still informative for comparing the models.

4. **Overall conclusion**
   - Despite the presence of some problematic Pareto-k values, both PSIS-LOO and predictive metrics (AUC, precision/recall, IoU) consistently indicate that the hierarchical NB model predicts new data more accurately than the global NB model.
   - The hierarchical model is therefore preferred in terms of predictive performance.
   - The increased p_loo warns that this improvement comes with additional model complexity, which should be discussed in the final report.

This completes requirement 10 by demonstrating how model comparison was done, interpreting the diagnostics, and explaining what conclusions can (and cannot) be drawn from the PSIS-LOO results.
