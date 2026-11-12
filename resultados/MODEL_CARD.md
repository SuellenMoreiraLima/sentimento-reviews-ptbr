---
language: pt
license: mit
base_model: neuralmind/bert-base-portuguese-cased
pipeline_tag: text-classification
tags:
  - sentiment-analysis
  - portuguese
  - bertimbau
metrics:
  - f1
  - accuracy
---

# BERTimbau: sentimento de avaliações de compra (pt-BR)

Classifica avaliações de compra em português como **positivo** ou **negativo**.
Fine-tuning do [BERTimbau base](https://huggingface.co/neuralmind/bert-base-portuguese-cased) com avaliações públicas da Olist.

## Como usar

```python
from transformers import pipeline

clf = pipeline("text-classification", model="suellenlima/bertimbau-sentimento-reviews-ptbr")
clf("A entrega chegou atrasada e a comida fria.")
```

## Dados

- Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- Rótulos fracos pela nota: 1-2 negativo, 4-5 positivo; nota 3 descartada.
- Duplicatas removidas; divisão estratificada 80/10/10 (seed 42).
- Tamanhos: treino 25,946, validação 3,243, teste 3,244.

## Treino

learning rate 2e-05, 3 épocas, batch 32, max_length 128,
warmup 0.1, weight decay 0.01; melhor época escolhida pela F1 macro de validação.

## Resultados (conjunto de teste)

| Métrica | Valor |
|---|---|
| F1 macro | 0.933 |
| Acurácia | 0.941 |

## Limitações

- Treinado em avaliações de **e-commerce**, não de restaurantes ou delivery de comida.
- Os rótulos vêm da nota, então há ruído (nota alta com texto negativo e vice-versa).
- Tem dificuldade com **ironia** ("parabéns pela demora") e textos mistos.
- Não classifica sentimento neutro.
- Não deve ser usado para decisões automáticas sobre pessoas sem revisão humana.
