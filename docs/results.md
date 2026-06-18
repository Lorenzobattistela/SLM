# Resultados

Os dois treinamentos usaram 4,1B tokens, 15.641 steps, batch efetivo 128 e validação com 10M tokens.

| Modelo | Validation loss final | Validation perplexity final |
| --- | ---: | ---: |
| SuperBPE | 3.5838 | 36.0103 |
| BBPE GPT-2 | 2.8304 | 16.9521 |

## SuperBPE

![Validation loss SuperBPE](assets/plot_images/superbpe_val_loss.png)
![Validation perplexity SuperBPE](assets/plot_images/superbpe_val_perplexity.png)
![Train loss SuperBPE](assets/plot_images/superbpe_train_loss.png)
![Learning rate SuperBPE](assets/plot_images/superbpe_lr.png)

## BBPE GPT-2

![Validation loss BBPE GPT-2](assets/plot_images/bbpe_val_loss.png)
![Validation perplexity BBPE GPT-2](assets/plot_images/bbpe_val_perplexity.png)
![Train loss BBPE GPT-2](assets/plot_images/bbpe_train_loss.png)
![Learning rate BBPE GPT-2](assets/plot_images/bbpe_lr.png)
