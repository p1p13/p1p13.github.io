---
layout: post
title:  "Bayesian Vs Frequentist Inference"
date:   2023-06-11 10:41:39 +0530
categories: Murphy Notes

---

In this post, we'll compare the two different methods of Statistical Inference - Bayesian and Frequentist.

## Statistical Inference

Statistical Inference is the process of modeling uncertainty about parameters using a probablilty distribution.
For example - parameter can be the the mean, $$\theta$$ of a probability distribution, P.

### Bayesian Statistics

In this field, we use posterior distribution to represent our uncertainty.
This implies that the parameter $$\theta$$ is considered an unknown random variable. We start with a prior distribution $$P(\theta)$$ and define a likelihood function $$P(\theta|D)$$, where D stands for some fixed dataset. For example - it can be a list of values drawn from a Probability distribution.
We then use Bayes to rule to calculate posterior as:

$$P(\theta|D) = \dfrac{P(\theta)P(\theta|D)}{P(D)}$$ 

### Frequentist Statistics

Here we don't treat the parameter as a random variable. 
We define an estimator which is like a function that takes data as argument and returns an estimate of the parameter.The uncertainty is induced from changing the data i.e. how the quantity estimated from data would change if the data were changed.
For example - let's denote the estimator by $$\hat{\theta} = \pi(D)$$ and true parameter by $$\theta^*$$.
We sample S different datasets each of size N, from some true model $$P(x|\theta^*)$$ to get
$$\tilde{D}^{(s)} = \{x_n \sim P(x_n|\theta^*) : n = 1 : N\} $$
Let's denote this by $$D^{(s)} \sim \theta^*$$. Now, apply the estimator to each $$D^{(s)}$$ to get a set of estimates $$\{\hat{\theta}(D^{(s)})\}$$.
The uncetainty then is represented by the following sampling distribution of the estimator:

$$P(\pi(\tilde{D}) = \theta | \tilde{D} \sim \theta^*) \approx \frac {\sum_{s=1}^{S}\delta(\theta=\pi(D^{(s)}))} {S}$$


### Credible Vs Confidence Intervals

Now let's discuss the differences between credible and confidence intervals. Credible interval is assosciated with bayesian statistics while confidence interval is a frequentist concept.

#### Credible Intervals

We usually use a point estimate to summarize a posterior distribution and then compute a credible interval to quantify the uncertainty associated with that estimate.

Precisely, we define a $$100(1-\alpha)\%$$ credible interval to be a contiguous region C = (l,u) which contains $$1-\alpha$$ of the posterior probability mass, i.e.,
$$C_\alpha(D) = (l,u): P(l \le \theta \le u | D) = 1 - \alpha$$

Since there are many intervals that satisfy this equation, we usually choose the one that allocates $$\frac {(1 - \alpha)} 2$$ mass in each tail and call this a central interval.

#### Confidence Intervals

We define a $$100(1 - \alpha)\%$$ confidence interval for a parameter $$\theta$$ as any interval $$I(\tilde{D}) = (l(\tilde{D}), u(\tilde{D}))$$ derived from a hypothetical dataset $$\tilde{D}$$ st

$$P(\theta \in I(\tilde{D}) |  \tilde{D} \sim \theta) = 1 - \alpha $$

This means if we repeatedly sampled data and compute $$I(\tilde{D})$$ for each dataset, about $$100(1-\alpha)\%$$  of such intervals will contain the true parameter $$\theta$$.


References:
==
https://probml.github.io/pml-book/book1.html

