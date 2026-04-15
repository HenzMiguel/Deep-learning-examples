# Simples treinamento e teste de CNN

## Foi utilizado o modelo resnet18 como base

- Best_model.pt contêm o modelo provindo do treinamento.
- Dataset utilizado foi [**Venomous and Non-venomous Animals**](https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india/data).
- A divisão feita foi train, eval e test. Com a pasta test sendo dividida em 80% treino e 20% eval para os hiperparâmetros
- Hiperparâmetros utilizados: Congelar a parte de extração de features ou não, tamanho dos batchs, learning rate e numero de epochs.
- Melhores hiperparâmetros estão em: best_params.json()


## Tecnologias utilizadas

- pytorch
- pytorch vision
- Optuna
- Matplotlib
- CNNs