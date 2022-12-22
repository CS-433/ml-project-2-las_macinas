"""
Compute metrics needed to evaluate our model
"""

import torch

def compute_metrics(confusion_vector, print_values=False):
  """
  Compute confusion matrix, precision, recall and return f1 score
  """
  # Compute confusion matrix values
  true_positives = torch.sum(confusion_vector == 1).item()
  false_positives = torch.sum(confusion_vector == float('inf')).item()
  true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
  false_negatives = torch.sum(confusion_vector == 0).item()

  # Print values if wanted
  if print_values == True:
    print('TP:', true_positives, '// FP:', false_positives, '//TN:', true_negatives, '//FN:', false_negatives)

  # Compute f1 score
  if ((true_positives+false_positives) != 0) and true_positives !=0: # division by 0
      precision_val = true_positives / (true_positives+false_positives)
      recall_val = true_positives / (true_positives+false_negatives)
      f1_val = 2*(precision_val*recall_val) / (precision_val+recall_val)
  else:
      f1_val = torch.tensor([[0.]]).item()

  return f1_val
