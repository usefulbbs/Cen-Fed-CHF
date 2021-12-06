# Cen-Fed-CHF
%% This is first version of the proposed Multiscale residule U++. The path in the code will be updated later. %%

Congestive heart failure (CHF), a progressive and complex syndrome caused by ventricular dysfunction, is difficult to detect in the early stage. Heart rate variability was proposed as a prognostic indicator for CHF. Inspired by the success of 2D UNet++ in medical image segmentation, in this paper, we introduce a multiscale residual encoder-decoder model for CHF detection based on short-term RR intervals. The main contributions are summarized as follows:

  1. This paper proposes a novel strategy to classify patients with CHF and various non-CHF subjects via improved 1-D UNet++ with multiscale fusion and residual connections. We concatenate edge units of UNet++ at different scales to increase the sensitivity to informative features.
  
  2.Considering the limited number of samples in each institution, we leverage the pruned variant of UNet++ with grouped convolution to adjust the model complexity and improve the performance. Based on the grouped convolution, both the number of parameters and the computational cost is significantly reduced. The improved UNet++ has achieved state-of-theart accuracy of 89.82% on two publicly available datasets with mild CHF patients.
  
  3.The proposed approach is implemented following the centralized learning and the federated learning framework. Based on extensive experiments on the publicly available datasets with mild CHF patients, we show that it is feasible to train the proposed CHF detection model distributively over multiple clients in a privacy-respecting manner.
