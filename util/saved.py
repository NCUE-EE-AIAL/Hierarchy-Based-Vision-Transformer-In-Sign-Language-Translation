import os
import torch

# List to store the best model file names and their validation losses
best_models = []


def save_best_models(model, bleu, step, save_dir='saved', max_models=3):
	global best_models

	# Create the directory if it doesn't exist
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# If we have fewer than `max_models` saved, or if the current loss is better than the worst saved loss
	if len(best_models) < max_models or bleu > max(best_models, key=lambda x: x[1])[1]:
		model_filename = f'model-{step}-{bleu:.4f}.pt'
		model_path = os.path.join(save_dir, model_filename)
		torch.save(model.state_dict(), model_path)

		best_models.append((model_filename, bleu))
		best_models = sorted(best_models, key=lambda x: x[1])  # Sort by loss in ascending order

		# If we have more than `max_models`, remove the worst model
		if len(best_models) > max_models:
			worst_model = best_models.pop(0)  # Remove the first (worst) model
			worst_model_path = os.path.join(save_dir, worst_model[0])
			if os.path.exists(worst_model_path):
				os.remove(worst_model_path)