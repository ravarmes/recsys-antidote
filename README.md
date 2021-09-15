<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-antidote/blob/master/assets/logo.jpg" />
</h1>

<h3 align="center">
  Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems
</h3>

<p align="center">Exemplo de medidas de justiça do usuário em Sistemas de Recomendação </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-antidote?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/scv-backend-spring/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-antidote?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-links">Links</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

> É proposto o desenvolvimento de um algoritmo com cálculos de medidas de justiça do usuário em Sistemas de Recomendação.

O objetivo deste repositório é implementar os cálculos de medidas de justiça do usuário propostas no artigo 'Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems' (WSDM 19)

Este repositório está baseada nas implementações do respositório [antidote-data-framework](https://github.com/rastegarpanah/antidote-data-framework) 

### Funções de Objetivo Social (Social Objective Functions)

```
* Polarization
* Individual fairness
* Group fairness
```

### Arquivos

```
* ArticleAntidoteData: implementação das medidas de justiça do usuário (ou funções de objetivo social)
* RecSysALS: implementação do sistema de recomendação baseado em filtragem colaborativa utilizando ALS (mínimos quadrados alternados)
* RecSysExampleData20Items: implementação de uma matriz de recomendações estimadas (apenas exemplo com valores aleatórios)
* TestArticleAntidoteData: arquivo para testar a implementação ArticleAntidoteData
```

## :link: Links <a name="-links"/></a>

- [Google Colaboratory](https://colab.research.google.com/drive/1aZIuljttlAaTq-LxtcXgjuBNnDCakuzE) - Notebook para demonstrar a utilização do algoritmo para uma base de dados pequena (40 usuário e 20 filmes);
- [Artigo](https://arxiv.org/pdf/1812.01504.pdf) - Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems;


## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: