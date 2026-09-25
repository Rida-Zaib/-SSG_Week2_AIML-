# Skill Set Go EduTech — AI/ML Track — Week 2

Model-Building Phase: Deep Learning with PyTorch

This repo contains all 4 tasks for Week 2 of the Skill Set Go EduTech AI/ML
internship track, merged from individual task repos into one place.

## Tasks

| Task | Folder | Description | Deliverable |
|---|---|---|---|
| 2.1 | [`2.1_neural-network-pytorch/`](./2.1_neural-network-pytorch) | Feed-forward neural network in PyTorch, covering tensors, datasets, dataloaders, and loss functions | PyTorch training notebook |
| 2.2 | [`2.2_cnn-image-classifier/`](./2.2_cnn-image-classifier) | CNN trained on the digits dataset to classify handwritten digit images | Trained CNN model + accuracy report |
| 2.3 | [`2.3_hyperparameter-experimentation/`](./2.3_hyperparameter-experimentation) | Controlled experiments varying optimizer, learning rate, batch size, and epochs | Experiment log with comparison table |
| 2.4 | [`2.4_deep-learning-report/`](./2.4_deep-learning-report) | Technical report comparing model performance across experiments | 1-2 page technical report (PDF) |

Each folder has its own `README.md` with setup and run instructions specific
to that task.

## Key results

| Model | Test Accuracy |
|---|---|
| Feed-forward NN (2.1) | 96.39% |
| CNN (2.2) | 96.39% |

Best hyperparameter configuration (2.3): Adam, lr=0.001, batch size 16, 10
epochs — 97.78% accuracy. Full comparison table and analysis in the task 2.4
report.

## How to use this repo

```bash
git clone https://github.com/Rida-Zaib/SSG_Week2_AIML.git
cd SSG_Week2_AIML
```

Then open whichever task folder you want and follow its own README.

## Progress

Week 2 complete — 8 of 17 total deliverables done (16 tasks + capstone across
the full 4-week AI/ML track).

## About

Part of the [Skill Set Go EduTech](https://skillsetgoedutech.vercel.app)
AI/ML Track — Learn. Build. Prove.
