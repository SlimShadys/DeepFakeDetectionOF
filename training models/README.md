# Results
The first tests showed that Optical Flow estimation wasn't performing as expected, thus approaching a validation accuracy for both ResNet-50 and VGG16 around 50-55%.<br>The following tables are referring to the new Optical Flow estimation.

## ResNet-50
Model &nbsp; &nbsp; | OF | Dataset | Optimizer | Epochs | Batch | LR | Val Acc | Link |
---|---|---|---|---|---|---|---|---|
| `ResNet-50` | PWC-Net | Real / F2F | Adam | 50 | 64 | 1e-4 | <b>71.40%</b> | [Link](https://drive.google.com/file/d/10AfkA5GW4wbwFz14OUO6VhTZp-nb2zXt/view?usp=drive_link) |
| `ResNet-50` | Raft | Real / Custom | Adam | 50 | 64 | 1e-4 | 62.28% | [Link](https://drive.google.com/file/d/10SMDRBZYD_p1CTQj3AmvXsjUPdcJ1Gq3/view?usp=drive_link) |
| `ResNet-50` | GMA | Real / Custom | Adam | 50 | 64 | 1e-4 | 64.65% | [Link](https://drive.google.com/file/d/1-pDTTCEXFcfeo3qm5oqPxxkqOvMOw8Rq/view?usp=drive_link) |

## VGG-16
Model &nbsp; &nbsp; | OF | Dataset | Optimizer | Epochs | Batch | LR | Val Acc | Link |
---|---|---|---|---|---|---|---|---|
| `VGG-16` | PWC-Net | Real / F2F | Adam | 50 | 64 | 1e-4 | <b>78.22%</b> | [Link](https://drive.google.com/file/d/10Cc9IC-FGCeLW_f75pJsP1xhwVjfLLl2/view?usp=drive_link) |
| `VGG-16` | Raft | Real / Custom | Adam | 50 | 64 | 1e-4 | 64.19% | [Link](https://drive.google.com/file/d/10Z2sFhxdte62XBvpNW4Yk1Cuq9OZPAVc/view?usp=drive_link) |
| `VGG-16` | GMA | Real / Custom | Adam | 50 | 64 | 1e-4 | 66.68% | [Link](https://drive.google.com/file/d/103XFqeeuwjZvB9RIdFZyp9YnHOoajMYh/view?usp=drive_link) |

## Vision Transformers
Model &nbsp; &nbsp; | OF | Dataset | Optimizer | Epochs | Batch | LR | Val Acc | Link |
---|---|---|---|---|---|---|---|---|
| `ViT-S/16` | PWC-Net | Real / F2F | AdamW | 50 | 64 | 1e-3 | 68.48% | [Link](https://drive.google.com/file/d/10EbIGoN0NOgaUw7KTzskpt_WZ8Z7h9so/view?usp=drive_link) |
| `DeiT-B/16` | PWC-Net | Real / F2F | AdamW | 50 | 64 | 1e-3 | <b>70.97%</b> | [Link](https://drive.google.com/file/d/109CKPLsphcoBP18RjbQR96U5aA-edn4T/view?usp=drive_link) |
| `ViT-S/16` | Raft | Real / Custom | AdamW | 50 | 64 | 1e-3 | 59.27% | [Link](https://drive.google.com/file/d/10a1g1Q3NA5KpY9HvUQxlasOKhKfSj9e9/view?usp=drive_link) |
| `DeiT-B/16` | Raft | Real / Custom | AdamW | 50 | 64 | 1e-3 | 60.51% | [Link](https://drive.google.com/file/d/10OXBvVyJDgpyEAtY_KEIxBFdmA1mzW46/view?usp=drive_link) |
| `ViT-S/16` | GMA | Real / Custom | AdamW | 50 | 64 | 1e-3 | 59.54% | [Link](https://drive.google.com/file/d/104FMJnzjjKHxGkGzWNGGkLvJoq7hKP4v/view?usp=drive_link) |
| `DeiT-B/16` | GMA | Real / Custom | AdamW | 50 | 64 | 1e-3 | 60.72% | [Link](https://drive.google.com/file/d/1-dKiftXf_voZnnjFlfIN3C3OEMVk8STl/view?usp=drive_link) |
