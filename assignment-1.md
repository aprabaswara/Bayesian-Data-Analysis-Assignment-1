
**University of Edinburgh**

**School of Mathematics**

**Bayesian Data Analysis, 2021/2022, Semester 2**

**Author: Aditya Prabaswara Mardjikoen (S2264710)**

**Lecturer: Daniel Paulin**

**Assignment 1**

**IMPORTANT INFORMATION ABOUT THE ASSIGNMENT**

**In this paragraph, we summarize the essential information about this
assignment. The format and rules for this assignment are different from
your other courses, so please pay attention.**

**1) Deadline: The deadline for submitting your solutions to this
assignment is the 7 March 12:00 noon Edinburgh time.**

**2) Format: You will need to submit your work as 2 components: a PDF
report, and your R Markdown (.Rmd) notebook. There will be two separate
submission systems on Learn: Gradescope for the report in PDF format,
and a Learn assignment for the code in Rmd format. You are encouraged to
write your solutions into this R Markdown notebook (code in R chunks and
explanations in Markdown chunks), and then select Knit/Knit to PDF in
RStudio to create a PDF report.**

<img src="knit_to_PDF.jpg" width="191" />

**It suffices to upload this PDF in Gradescope submission system, and
your Rmd file in the Learn assignment submission system. You will be
required to tag every sub question on Gradescope. A video describing the
submission process will be posted on Learn.**

**Some key points that are different from other courses:**

**a) Your report needs to contain written explanation for each question
that you solve, and some numbers or plots showing your results.
Solutions without written explanation that clearly demonstrates that you
understand what you are doing will be marked as 0 irrespectively whether
the numerics are correct or not.**

**b) Your code has to be possible to run for all questions by the Run
All in RStudio, and reproduce all of the numerics and plots in your
report (up to some small randomness due to stochasticity of Monte Carlo
simulations). The parts of the report that contain material that is not
reproduced by the code will not be marked (i.e. the score will be 0),
and the only feedback in this case will be that the results are not
reproducible from the code.**

<img src="run_all.jpg" width="376" />

**c) Multiple Submissions are allowed BEFORE THE DEADLINE are allowed
for both the report, and the code. However, multiple submissions are NOT
ALLOWED AFTER THE DEADLINE. YOU WILL NOT BE ABLE TO MAKE ANY CHANGES TO
YOUR SUBMISSION AFTER THE DEADLINE. Nevertheless, if you did not submit
anything before the deadline, then you can still submit your work after
the deadline. Late penalties will apply unless you have a valid
extension. The timing of the late penalties will be determined by the
time you have submitted BOTH the report, and the code (i.e. whichever
was submitted later counts).**

**We illustrate these rules by some examples:**

**Alice has spent a lot of time and effort on her assignment for BDA.
Unfortunately she has accidentally introduced a typo in her code in the
first question, and it did not run using Run All in RStudio. - Alice
will get 0 for the whole assignment, with the only feedback “Results are
not reproducible from the code”.**

**Bob has spent a lot of time and effort on his assignment for BDA.
Unfortunately he forgot to submit his code. - Bob will get no personal
reminder to submit his code. Bob will get 0 for the whole assignment,
with the only feedback “Results are not reproducible from the code, as
the code was not submitted.”**

**Charles has spent a lot of time and effort on his assignment for BDA.
He has submitted both his code and report in the correct formats.
However, he did not include any explanations in the report. Charles will
get 0 for the whole assignment, with the only feedback “Explanation is
missing.”**

**Denise has spent a lot of time and effort on her assignment for BDA.
She has submitted her report in the correct format, but thought that she
can include her code as a link in the report, and upload it online (such
as Github, or Dropbox). - Denise will get 0 for the whole assignment,
with the only feedback “Code was not uploaded on Learn.”**

**3) Group work: This is an INDIVIDUAL ASSIGNMENT, like a 2 week exam
for the course. Communication between students about the assignment
questions is not permitted. Students who submit work that has not been
done individually will be reported for Academic Misconduct, that can
lead to serious consequences. Each problem will be marked by a single
instructor, so we will be able to spot students who copy.**

**4) Piazza: You are NOT ALLOWED to post questions about Assignment
Problems visible to Everyone on Piazza. You need to specify the
visibility of such questions as Instructors only, by selecting Post to /
Individual students/Instructors and type in Instructors and click on the
blue Instructors banner that appears below**

![](piazza_instructors.jpg)

**Students who post any information related to the solution of
assignment problems visible to their classmates will**

**a) have their access to Piazza revoked for the rest of the course
without prior warning, and**

**b) reported for Academic Misconduct.**

**Only questions regarding clarification of the statement of the
problems will be answered by the instructors. The instructors will not
give you any information related to the solution of the problems, such
questions will be simply answered as “This is not about the statement of
the problem so we cannot answer your question.”**

**THE INSTRUCTORS ARE NOT GOING TO DEBUG YOUR CODE, AND YOU ARE ASSESSED
ON YOUR ABILITY TO RESOLVE ANY CODING OR TECHNICAL DIFFICULTIES THAT YOU
ENCOUNTER ON YOUR OWN.**

**5) Office hours: There will be two office hours per week (Monday
16:00-17:00, and Wednesdays 16:00-17:00) during the 2 weeks for this
assignment. The links are available on Learn / Course Information. We
will be happy to discuss the course/workshop materials. However, we will
only answer questions about the assignment that require clarifying the
statement of the problems, and will not give you any information about
the solutions. Students who ask for feedback on their assignment
solutions during office hours will be removed from the meeting.**

**6) Late submissions and extensions: Students who have existing
Learning Adjustments in Euclid will be allowed to have the same
adjustments applied to this course as well, but they need to apply for
this BEFORE THE DEADLINE on the website**

<https://www.ed.ac.uk/student-administration/extensions-special-circumstances>

**by clicking on “Access your learning adjustment”. This will be
approved automatically.**

**For students without Learning Adjustments, if there is a justifiable
reason (external circumstances) for not being able to submit your
assignment in time, then you can apply for an extension BEFORE THE
DEADLINE on the website**

<https://www.ed.ac.uk/student-administration/extensions-special-circumstances>

**by clicking on “Apply for an extension”. Such extensions are processed
entirely by the central ESC team. The course instructors have no role in
this decision so you should not write to us about such applications. You
can contact our Student Learning Advisor, Maria Tovar Gallardo
(<maria.tovar@ed.ac.uk>) in case you need some advice regarding this.**

**Students who submit their work late will have late submission
penalties applied by the ESC team automatically (this means that even if
you are 1 second late because of your internet connection was slow, the
penalties will still apply). The penalties are 5% of the total mark
deduced for every day of delay started (i.e. one minute of delay counts
for 1 day). The course intructors do not have any role in setting these
penalties, we will not be able to change them.**

<img src="rotifer.jpg" style="width:56.0%" />

<img src="algae.jpg" style="width:38.0%" />

The first picture is a rotifier (by Steve Gschmeissner), the second is a
unicellular algae (by NEON ja, colored by Richard Bartz).

**Problem 1 - Rotifier and algae data**

**In this problem, we study an experimental dataset (Blasius et al.
2020, <https://doi.org/10.1038/s41586-019-1857-0>) about predator-prey
relationship between two microscopic organism: rotifier (predator) and
unicellular green algae (prey). These were studied in a controlled
environment (water tank) in a laboratory over 375 days. The dataset
contains daily observations of the concentration of algae and rotifiers.
The units of measurement in the algae column is** $\mathbf{10^6}$
**algae cells per ml of water, while in the rotifier column it is the
number of rotifiers per ml of water.**

**We are going to apply a simple two dimensional state space model on
this data using JAGS. The first step is to load JAGS and the dataset.**

``` r
# We load JAGS
library(rjags)
```

    ## Loading required package: coda

    ## Linked to JAGS 4.3.1

    ## Loaded modules: basemod,bugs

``` r
#You may need to set the working directory first before loading the dataset
#setwd("/Users/dpaulin/Dropbox/BDA_2021_22/Assignments/Assignment1")
rotifier_algae=read.csv("rotifier_algae.csv")
#The first 6 rows of the dataframe
print.data.frame(rotifier_algae[1:6,])
```

    ##   day algae rotifier
    ## 1   1  1.50       NA
    ## 2   2  0.82     6.58
    ## 3   3  0.77    17.94
    ## 4   4  0.36    17.99
    ## 5   5  0.41    21.12
    ## 6   6  0.41    17.06

**As we can see, some values in the dataset are missing (NA)**.

**We are going to model the true log concentrations** $\mathbf{x}_t$
**by the state space
model**$$\mathbf{x}_t = \mathbf{A} \mathbf{x}_{t-1}+\mathbf{b}+\mathbf{w}_t; \quad \mathbf{w}_t\sim N\left(0,\left(\begin{matrix}\sigma_R^2 & 0\\ 0 & \sigma_A^2\end{matrix}\right)\right)$$  
**where** $\mathbf{A}$**,** $\mathbf{b}$, $\sigma^2_R$ **and**
$\sigma^2_A$ **are model parameters, and** $t$ **denotes the time point.
In particular,** $t=0$ **corresponds to day 0, and** $t=1,2,\ldots, 375$
**correspond to days 1-375.**

**Here** $\mathbf{x}_t$ **is a two dimensional vector. The first
component denotes the logarithm of the rotifier concentration measured
in number of rotifiers per ml of water, and the second component denotes
the logarithm of the algae concentration measured in** $10^6$ **algae
per ml (these units are the same as in the dataset).**
$\mathbf{A}=\left(\begin{matrix}A_{11} & A_{12}\\ A_{21} & A_{22}\end{matrix}\right)$
**is a two times two matrix, and** $\mathbf{b}$ **is a two dimensional
vector.**

**The observation process is described as**
$$\mathbf{y}_t =\mathbf{x}_{t}+\mathbf{v}_t, \quad \mathbf{v}_t\sim N\left(0,\left(\begin{matrix}\eta_R^2 & 0\\ 0 & \eta_A^2\end{matrix}\right)\right),$$

**where** $\mathbf{y}_t$ **is the observed log concentration on day**
$t$ **(for example,**
$\mathbf{y}_2=\left(\begin{matrix}\log(6.58)\\ \log(0.82)\end{matrix}\right)$
**in our dataset), while** $\eta_R^2$ **and** $\eta_R^2$ **are
additional model parameters.**

**a)\[10 marks\] Create a JAGS model that fits the above state space
model on the rotifier-algae dataset for the whole 375 days period.**

**Use 10000 burn-in steps and obtain 50000 samples from the model
parameters**
$\mathbf{A}, \mathbf{b}, \sigma_R^2, \sigma_A^2, \eta_R^2, \eta_A^2$
**(4+2+4=10 parameters in total).**

**Use a Gaussian prior**
$N\left(\left(\begin{matrix}\log(6)\\ \log(1.5) \end{matrix}\right), \left(\begin{matrix}4 & 0\\ 0 & 4\end{matrix}\right) \right)$
**for the initial state** $\mathbf{x}_0$**, independent Gaussian**
$N(0,1)$ **priors for each 4 elements of** $\mathbf{A}$, **Gaussian
prior**
$N\left(\left(\begin{matrix}0\\ 0 \end{matrix}\right), \left(\begin{matrix}1 & 0\\ 0 & 1\end{matrix}\right) \right)$
**for** $\mathbf{b}$**, and inverse Gamma (0.1,0.1) prior for the
variance parameters** $\sigma_R^2, \sigma_A^2, \eta_R^2, \eta_A^2$**.**

**Explain how did you handle the fact that some of the observations are
missing (NA) in the dataset.**

From the first 6 rows of the rotifier-algae dataset we can see that the
data contain some missing values. For instance, in the first
observations in the first row we can see that the concentration of algae
is observed but the concentration of rotifier is missing. This can be
known as a partially observed and partially missing pair of observations
in the dataset. Since we use a multivariate normal distribution with two
dimensional vector in our jags model especially in the observed log of
concentration ($\mathbf{y}_t$) in our state space model in JAGS, we
can’t have a partially observed and partially missing observations
because it can’t make JAGS output an error when running our state space
model. This is somehow different with the case of univariate data with
missing values, where in this type of univariate data JAGS will still
generate samples from their posterior predictive distributions without
giving output error in computer even though the input data contain
missing values. Therefore, we have to deal with the missing values first
before we input our data into our JAGS model. Since this is a time
series data, we are not allowed to omit data points with $\texttt{NA}$
values because this would move all later time points backwards.
Moreover, we can’t impute the missing value with some interpolated or
arbitary values since this would change the posterior distribution of
the log of concentration. Thus, I would prefer to use the property of
multivariate normal distribution in order to specify the distribution of
each component of our multivariate vector, which in this case is
$\mathbf{y}_t$. Assume $\mathbf{y}_t = (y_{1t}, y_{2t})^\mathbf{T}$,
where $y_{1t}$ and $y_{2t}$ are the observed log-concentration of
rotifier and algae at days $t$ respectively. In addition, assume too
that $\mathbf{x}_t = (x_{1t}, x_{2t})^\mathbf{T}$ where $x_{1t}$ and
$x_{2t}$ are the true observed log-concentration of rotifier and algae
respectively. Recall that if an arbitrary vector
$\mathbf{X} = \left(X_1,\dots,X_n\right)^\mathbf{T}$ is distributed as
$N\left(\pmb{\mu},\Sigma\right)$ where $\pmb{\mu}$ and
$$\Sigma = \left(\begin{matrix}\sigma_{11} & \sigma_{12}& \cdots & \sigma_{1n}\\ \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2n}\\ \vdots & \vdots & \cdots & \vdots\\ \sigma_{n1} & \sigma_{n2} & \cdots & \sigma_{nn}\end{matrix}\right)$$
is the mean vector and covariance matrix respectively, then any linear
combinations of variables
$\pmb{a}^\mathbf{T}\pmb{X} = a_1X_1+a_2X_2+\dots+a_nX_n$ is distributed
as
$N\left(\pmb{a}^\mathbf{T}\pmb{\mu},\pmb{a}^\mathbf{T}\Sigma\pmb{a}\right).$
Also, if $a^\mathbf{T}\mathbf{X}$ is distributed as
$N\left(\pmb{a}^\mathbf{T}\pmb{\mu},\pmb{a}^\mathbf{T}\Sigma\pmb{a}\right)$
for every $\pmb{a}$, then $\mathbf{X}$ must be distributed as
$N\left(a^\mathbf{T}\pmb{\mu}, a^\mathbf{T}\Sigma\pmb{a}\right)$. This
property can be found in Richard A. Johnson and Dean W. Wichern book
that title is “Applied Multivariate Statistical Analysis Sixth Edition”
in page 156. According to this property, if $\pmb{e_1},\dots,\pmb{e_n}$
is a standard basis vector in $\mathbb{R}^n$ (i.e. the $i$-th component
is 1 while others 0, for instance $e_1 = (1,0,\dots,0)^\mathbf{T}$),
then $X_i = e_i^\mathbf{T}\mathbf{X}$ is distributed as
$N\left(\mu_i,\sigma_{ii}\right)$. Since the main diagonal of a
covariance matrix is the variance, then $\sigma_{ii}$ is the variance in
our normal distribution parameter for random variable $X_i$. Now observe
that from our state space model of $\mathbf{y}_t$ we will have
$\mathbf{y}_t\sim N\left(\mathbf{x}_t,\left(\begin{matrix}\eta^2_R& 0\\ 0 & \eta^2_A\end{matrix}\right)\right)$.
Thus, we will have $y_{1t}$ and $y_{2t}$ are distributed as
$N\left(x_{1t},\eta^2_R\right)$ and $N\left(x_{2t},\eta^2_A\right)$
respectively. By specifying the prior distribution of each $y_{1t}$ and
$y_{2t}$ in JAGS using $\verb|dnorm|$ instead of specifying the prior
distribution of $\mathbf{y}_t$ using $\verb|dmnorm|$, JAGS won’t output
an error when we run our model since we input a univariate data for each
node of the component of vector $\mathbf{y}_t$. After we figure out a
solution to handle the missing data in JAGS, we will extend the data by
adding a pair of $\verb|NA|$ observations for rotifier and algae
concentration as our observations in day 0 due to our actual data
observations start from day 1 and in our state space model we specify
prior distribution for the true observed log of concentration for
rotifier and algae at day 0, which in this case is $\mathbf{x}_0$.
Lastly, we run our JAGS model with 10000 burn-in steps and get 50000
samples of the parameter
$\mathbf{A}, \mathbf{b}, \sigma_R^2, \sigma_A^2, \eta_R^2, \eta_A^2$. In
this report, I will use 9 chains to obtain the sample for our parameter
of interest. Due to $\mathbf{A}$ and $\mathbf{b}$ is $2\times2$ matrix
and two dimensional vector respectively, then we would have 10
parameters in total because JAGS will generate sample for each component
in $\mathbf{A}$ and $\mathbf{b}$ separately in a different nodes. One
important thing to highlight here is the parameter that were given for
the multivariate normal distribution in this problem is its mean vector
and covariance matrix, while in JAGS the parameter of the multivariate
normal distribution using $\verb|dmnorm|$ is in forms of mean vector and
precision matrix. Therefore, we should compute the precision matrix by
finding the inverse of the covariance matrix.

``` r
#We create the model string in JAGS
model_string <-   
  "model {
  # prior on the initial true log-concentration at day 0 (x_0), denoted by x[1] in R
  
  x[1,1:2] ~ dmnorm(mu.x0[1:2], prec.x0[1:2, 1:2])
  
  #prior for vector b
  b[1:2] ~ dmnorm(mu.b[1:2], prec.b[1:2, 1:2])
  
  #create matrix A
  for (i in 1:2) {
     for (j in 1:2){
        A[i,j] ~ dnorm(0, 1)
     }
  }
  
  #inverse gamma prior for each sigma square and eta square
  tau.A1 ~ dgamma(0.1, 0.1)
  tau.R1 ~ dgamma(0.1, 0.1)
  tau.A2 ~ dgamma(0.1, 0.1)
  tau.R2 ~ dgamma(0.1, 0.1)
  
  sigma2.A <- 1/tau.A1
  sigma2.R <- 1/tau.R1
  eta2.A <- 1/tau.A2
  eta2.R <- 1/tau.R2
  
  #covariance matrix for vector w
  Sigma.w[1,1] <- sigma2.R
  Sigma.w[1,2] <- 0
  Sigma.w[2,1] <- 0
  Sigma.w[2,2] <- sigma2.A
  
  #precision matrix for vector w
  prec.w <- inverse(Sigma.w)
  
  #covariance matrix for vector v
  Sigma.v[1,1] <- eta2.R
  Sigma.v[1,2] <- 0
  Sigma.v[2,1] <- 0
  Sigma.v[2,2] <- eta2.A
  
  #precision matrix for vector v
  prec.v <- inverse(Sigma.v)
  
  #Likelihood
  #n denotes the total number of days considered starting from day 0
  #i = 1 denotes day 0, i = 2 denotes day 1,.....
  for(i in 2:n) {
    #true observed log concentration
    x[i,1:2] ~ dmnorm(A %*% x[i-1,1:2]+b[1:2],prec.w[1:2,1:2])
  }
  
  for(i in 1:n) {
    #observed log concentration
    y1[i] ~ dnorm(x[i,1],prec.v[1,1])
    y2[i] ~ dnorm(x[i,2],prec.v[2,2])
    
    #replicate observed log concentration
    y1rep[i] ~ dnorm(x[i,1],prec.v[1,1])
    y2rep[i] ~ dnorm(x[i,2],prec.v[2,2])
    
  }
  
}"

#get last day and first day in our data
first_day <- min(rotifier_algae[['day']], na.rm = T)
last_day <- max(rotifier_algae[['day']], na.rm = T)

#number of days starting from day 0
n <- last_day-(first_day-1)+1

#get logarithm of rotifier and algae concentration
#y1 = rotifier, y2 = algae
y1 <-rep(NA,n)
y2 <-rep(NA,n)

for(i in 1:nrow(rotifier_algae)){
  
  y1[rotifier_algae[i,1]-(-1)]=log(rotifier_algae[i,3])
  y2[rotifier_algae[i,1]-(-1)]=log(rotifier_algae[i,2])
}

#input data and hyperparameter to our JAGS model
#n = number of days from day 0 until last day in our data
#prec.x0 = precision matrix for x_0
#prec.b = precision matrix for b
#mu.b = mean vector for b
#mu.x0 = mean vector for x_0

model.data <- list(mu.x0 = c(log(6), log(1.5)),mu.b = c(0,0),
                   prec.x0 = solve(diag(4,2)),y1 = y1,y2 = y2, 
                   prec.b = solve(diag(1,2)),
                   n = n)

model <- jags.model(textConnection(model_string), n.chains = 9, 
                    data = model.data)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 735
    ##    Unobserved stochastic nodes: 1154
    ##    Total graph size: 3422
    ## 
    ## Initializing model

``` r
#Burnin for 10000 samples
update(model,10000,progress.bar="none")

#Generate MCMC sample for A, b, eta square and sigma square
sample_param <- c("A","b","sigma2.A","eta2.A","sigma2.R","eta2.R")

res.model <- coda.samples(model,variable.names = sample_param, 
                          n.iter=50000,progress.bar="none")
```

**b)\[10 marks\] Based on your MCMC samples, compute the Gelman-Rubin
convergence diagnostics (Hint: you need to run multiple chains in
parallel for this by setting the n.chains parameter). Discuss how well
has the chain converged to the stationary distribution based on the
results.**

**Print out the summary of the fitted JAGS model. Do autocorrelation
plots for the 4 components of the model parameter** $\mathbf{A}$**.**

**Compute and print out the effective sample sizes (ESS) for each of the
model parameters**
$\mathbf{A}, \mathbf{b}, \sigma_R^2, \sigma_A^2, \eta_R^2, \eta_A^2$**.**

**If the ESS is below 1000 for any of these 10 parameters, increase the
sample size/number of chains until the ESS is above 1000 for all 10
parameters.**

First lets display the summary statistics for our $\mathbf{A}$,
$\mathbf{b}$, $\sigma^2_A$, $\sigma^2_R$, $\eta^2_A$ and $\eta^2_R$ from
our JAGS sample. In the summary statistics of our fitted JAGS model, we
are given the value of the mean, standard deviation and various quantile
of the posterior distribution of $\sigma^2_A$, $\sigma^2_R$, $\eta^2_A$,
$\eta^2_R$ and each component of $\mathbf{A}$ and $\mathbf{b}$. The 2.5%
and 97.5% quantile in the summary statistics given in the $\verb|R|$
code output below is the upper and lower bound of the 95% credible
interval for each ten parameter in our MCMC sample. In Bayesian
statistics, we can interpret the 95% credible interval as a fix interval
where the value of interest of our parameter will lies on that interval
with a 95% probability.

``` r
#Display some summary statistics
summary(res.model)
```

    ## 
    ## Iterations = 11001:61000
    ## Thinning interval = 1 
    ## Number of chains = 9 
    ## Sample size per chain = 50000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##              Mean       SD  Naive SE Time-series SE
    ## A[1,1]    0.57447 0.044572 6.644e-05      7.199e-04
    ## A[2,1]   -0.42153 0.048619 7.248e-05      7.550e-04
    ## A[1,2]    0.59867 0.087066 1.298e-04      1.252e-03
    ## A[2,2]    0.25673 0.058302 8.691e-05      3.340e-04
    ## b[1]      1.84724 0.137910 2.056e-04      2.130e-03
    ## b[2]      0.73257 0.158008 2.355e-04      2.470e-03
    ## eta2.A    0.08961 0.030304 4.517e-05      4.865e-04
    ## eta2.R    0.02705 0.008311 1.239e-05      8.558e-05
    ## sigma2.A  0.16687 0.032021 4.773e-05      4.600e-04
    ## sigma2.R  0.13008 0.019173 2.858e-05      2.178e-04
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##              2.5%      25%      50%     75%    97.5%
    ## A[1,1]    0.48827  0.54433  0.57401  0.6041  0.66401
    ## A[2,1]   -0.51890 -0.45376 -0.42079 -0.3886 -0.32851
    ## A[1,2]    0.45110  0.53679  0.59082  0.6521  0.79026
    ## A[2,2]    0.14026  0.21803  0.25746  0.2960  0.36945
    ## b[1]      1.57903  1.75459  1.84643  1.9389  2.12085
    ## b[2]      0.42991  0.62558  0.73043  0.8381  1.04651
    ## eta2.A    0.03513  0.06752  0.08846  0.1102  0.15128
    ## eta2.R    0.01376  0.02102  0.02607  0.0320  0.04595
    ## sigma2.A  0.10810  0.14400  0.16582  0.1886  0.23158
    ## sigma2.R  0.09152  0.11748  0.13037  0.1430  0.16701

Next, we will analyze the trace plot for each parameters. From the trace
plot below, we can see that the chains for each component of
$\mathbf{A}$, $\mathbf{b}$, $\sigma^2_A$, $\sigma^2_R$, $\eta^2_A$ and
$\eta^2_R$ were mixing well and have converged.

``` r
#trace plot and density plot for each b
plot(res.model[,c("b[1]","b[2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
#trace plot and density plot for each A
plot(res.model[,c("A[1,1]","A[1,2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
plot(res.model[,c("A[2,1]","A[2,2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

``` r
#trace plot and density plot for each sigma square
plot(res.model[,c("sigma2.A","sigma2.R")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->

``` r
#trace plot and density plot for each eta square
plot(res.model[,c("eta2.A","eta2.R")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-4-5.png)<!-- -->

The Gelman-Rubin statistics and plots are shown below. We also can see
that our Gelman-Rubin statistics for all ten parameter in our sample is
equal to 1. Therefore, we can say that the chains for all ten parameter
in our MCMC sample have converged as the Gelman-Rubin plot and trace
plot suggested. Lets have a look at the autocorrelation plot for each
components of $\mathbf{A}$. We can see that the autocorrelation for each
chain in each components of $\mathbf{A}$ is always decreasing faster
when the lag keep increasing. This indicate that every current and
previous sample for each chain in each component of $\mathbf{A}$ are not
highly correlated as the lags increase.

``` r
#autocorrelation plot for A
acfplot(res.model[,c("A[1,1]","A[1,2]","A[2,1]","A[2,2]")], 
        lag.max = 100, aspect=1, type = 'l', ylim = c(0,1))
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
#Gelma-Rubin plot for each b
par(mfrow=c(1,2))
gelman.plot(res.model[,c("b[1]","b[2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

``` r
#Gelma-Rubin plot for each A
gelman.plot(res.model[,c("A[1,1]","A[1,2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

``` r
gelman.plot(res.model[,c("A[2,1]","A[2,2]")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-4.png)<!-- -->

``` r
#Gelma-Rubin plot for each sigma square
gelman.plot(res.model[,c("sigma2.A","sigma2.R")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-5.png)<!-- -->

``` r
#Gelma-Rubin plot for each eta square
gelman.plot(res.model[,c("eta2.A","eta2.R")])
```

![](assignment-1_files/figure-gfm/unnamed-chunk-5-6.png)<!-- -->

``` r
#Gelman-Rubin statistics
gelman.diag(res.model)
```

    ## Potential scale reduction factors:
    ## 
    ##          Point est. Upper C.I.
    ## A[1,1]            1       1.01
    ## A[2,1]            1       1.01
    ## A[1,2]            1       1.01
    ## A[2,2]            1       1.00
    ## b[1]              1       1.01
    ## b[2]              1       1.01
    ## eta2.A            1       1.01
    ## eta2.R            1       1.00
    ## sigma2.A          1       1.00
    ## sigma2.R          1       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1.01

Now lets check for the effective sample size for each parameter. We can
see that the effective sample size for each ten parameter in our MCMC
sample are greater than 1000. This implies that the effective sample
size of our ten parameter in our MCMC sample is very large,
corresponding to good mixing in each chain as the trace plot for each
ten parameter shown.

``` r
#effective sample sizes
cat("Effective sample size for each parameter:\n")
```

    ## Effective sample size for each parameter:

``` r
effectiveSize(res.model)
```

    ##    A[1,1]    A[2,1]    A[1,2]    A[2,2]      b[1]      b[2]    eta2.A    eta2.R 
    ##  3849.505  4143.444  4875.550 30602.378  4208.445  4087.475  3922.128  9466.846 
    ##  sigma2.A  sigma2.R 
    ##  4922.316  7877.948

**c)\[10 marks\] We are going to perform posterior predictive checks to
evaluate the fit of this model on the data (using the priors stated in
question a). First, create replicate observations from the posterior
predictive using JAGS. The number of replicate observations should be at
least 1000.**

**Compute the minimum, maximum, and median for both log-concentrations
(i.e. both for rotifier and algae,** $3\cdot 2=6$ **in total).**

**Plot the histograms for these quantities together with a line that
shows the value of the function considered on the actual dataset (see
the R code for Lecture 2 for an example). Compute the DIC score for the
model (Hint: you can use the `dic.samples` function for this).**

**Discuss the results.**

The histograms below show posterior predictive distribution for the
minimum (top), maximum (middle) and median(bottom) log concentration of
rotifier (all plot in the left) and algae (all plot in the right) for
the Bayesian fit to the concentration of rotifier and algae time series
data. The descriptive statistic values (minimum, maximum and median) in
the observed data set for the log concentration of rotifier and algae
are shown by vertical red lines. The lines seem to be within the typical
range of the replicates, showing a reasonably good fit. We can see that
the shape of predictive distribution for the descriptive statistic
values (minimum, maximum and median) of the replicates of log
concentration of rotifier all shaped like a normal distribution curve,
while for the replicates of log concentration of algae only the
histogram related to the predictive distribution of the median that
shaped like a normal distribution. The predictive distribution of the
replicates of log concentration of algae for the minimum and maximum
values are shaped like a left-skewed an right-skewed distribution curve
respectively.

``` r
library(fBasics)

#observed y1 and y2 data in JAGS sample
#rot_obs = observed y1, alg_obs = observed y2
#observed data is a data which is not NA in our original data

rot_obs <- which(!is.na(y1)) 
alg_obs <- length(y2)+which(!is.na(y2))

#generate replicate sample for y1 and y2
yrep.res <- coda.samples(model,n.iter=10000,
                         progress.bar="none",
                         variable.names=c("y1rep","y2rep"))

#store replicate sample for y1 and y2 in a data frame
yrep.rot <- data.frame(rbind(yrep.res[[1]][,rot_obs],
                             yrep.res[[2]][,rot_obs],
                             yrep.res[[3]][,rot_obs],
                             yrep.res[[4]][,rot_obs],
                             yrep.res[[5]][,rot_obs],
                             yrep.res[[6]][,rot_obs],
                             yrep.res[[7]][,rot_obs],
                             yrep.res[[8]][,rot_obs],
                             yrep.res[[9]][,rot_obs]))

yrep.alg <- data.frame(rbind(yrep.res[[1]][,alg_obs],
                             yrep.res[[2]][,alg_obs],
                             yrep.res[[3]][,alg_obs],
                             yrep.res[[4]][,alg_obs],
                             yrep.res[[5]][,alg_obs],
                             yrep.res[[6]][,alg_obs],
                             yrep.res[[7]][,alg_obs],
                             yrep.res[[8]][,alg_obs],
                             yrep.res[[9]][,alg_obs]))

#compute mean, minimum, and maximum value of replicate y1
yrep.rot.min <- apply(yrep.rot,MARGIN=1, FUN=min)
yrep.rot.max <- apply(yrep.rot,MARGIN=1, FUN=max)
yrep.rot.median <- apply(yrep.rot,MARGIN=1, FUN=median)

#compute mean, minimum, and maximum value of replicate y2
yrep.alg.min <- apply(yrep.alg,MARGIN=1, FUN=min)
yrep.alg.max <- apply(yrep.alg,MARGIN=1, FUN=max)
yrep.alg.median <- apply(yrep.alg,MARGIN=1, FUN=median)

#plot the predictive distribution histogram
par(mfrow=c(3,2))
hist(yrep.rot.min,col="gray40", cex.main = 0.75, 
     xlab ='min rotifier',
     main="Rotifier Predictive Distribution for Min")
abline(v=min(y1, na.rm = T),col="red",lwd=2)

hist(yrep.alg.min,col="gray40", cex.main = 0.75,
     xlab = 'min algae',
     main="Algae Predictive Distribution for Min")
abline(v=min(y2, na.rm = T),col="red",lwd=2)

hist(yrep.rot.max,col="gray40", cex.main = 0.75,
     xlab = 'max rotifier',
     main="Rotifier Predictive Distribution for Max")
abline(v=max(y1, na.rm = T),col="red",lwd=2)

hist(yrep.alg.max,col="gray40", cex.main = 0.75,
     xlab = 'min algae',
     main="Algae Predictive Distribution for Max")
abline(v=max(y2, na.rm = T),col="red",lwd=2)

hist(yrep.rot.median,col="gray40", cex.main = 0.75,
     xlab='median rotifier',
     main="Rotifier Predictive Distribution for Median")
abline(v=median(y1, na.rm = T),col="red",lwd=2)


hist(yrep.alg.median,col="gray40", cex.main = 0.75,
     xlab = 'median algae',
     main="Algae Predictive Distribution for Median")
abline(v=median(y2, na.rm = T),col="red",lwd=2)
```

![](assignment-1_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

The DIC score for our JAGS model that we create for 1a) is presented
below.

``` r
#DIC score for our JAGS model
dic.samples(model, n.iter=10000)
```

    ## Mean deviance:  -164.7 
    ## penalty 597 
    ## Penalized deviance: 432.2

**d)\[10 marks\] Discuss the meaning of the** **model parameters**
$\mathbf{A}, \mathbf{b}, \sigma_R^2, \sigma_A^2, \eta_R^2, \eta_A^2$**.
Find a website or paper that that contains information about rotifiers
and unicellular algae (Hint: you can use Google search for this). Using
your understanding of the meaning of model parameters and the biological
information about these organisms, construct more informative prior
distributions for the model parameters. State in your report the source
of information and the rationale for your choices of priors.**

**Re-implement the JAGS model with these new priors. Perform the same
posterior predictive checks as in part c) to evaluate the fit of this
new model on the data.**

**Compute the DIC score for the model as well (Hint: lower DIC score
indicates better fit on the data).**

**Discuss whether your new priors have improved the model fit compared
to the original prior from a).**

First, lets talk about the interpretation of parameter $\mathbf{A}$ and
$\mathbf{b}$. In our state space model, the parameter $\mathbf{b}$
represents the number of new organism (rotifier and algae) vector where
the first and second component is the number of new rotifier (measured
in number of new rotifiers per ml of water) and the number of new algae
(measured in $10^6$ cells per ml of water) respectively. In terms of
predator prey model, the matrix $\mathbf{A}$ represents the rate of
change in the log concentration of rotifier and algae. The component of
matrix $\mathbf{A}$ in the first column ($\mathbf{A}_{11}$ and
$\mathbf{A}_{21}$) of matrix $\mathbf{A}$ represents the rate of change
in the log concentration of rotifier, while the component in the second
column ($\mathbf{A}_{12}$ and $\mathbf{A}_{22}$) represents the rate of
change in the log concentration of algae.

Now lets talk about the interpretation of other parameters in our state
space model such as $\sigma^2_R$, $\sigma^2_A$, $\eta^2_A$, and
$\eta^2_R$. The parameter $\sigma^2_R$ and $\sigma^2_A$ represents the
variance of the true log concentration ($x_t$) of rotifier and algae
respectively. Similarly, the parameter $\eta^2_R$ and $\eta^2_A$
represents the variance of the observed log-concentration ($y_t$) of
rotifier and algae respectively.

After we give an interpretation about our model parameters, now we will
update our prior according to the information about rotifier and algae
which related to the model parameter. First, we will determine the new
prior for the model parameter $\mathbf{A}$, which represent the rate of
change in the log concentration of rotifier and algae. Recall that
rotifier eats algae. This implies that if the concentration of algae
increased then the concentration of rotifier will also increased since
the number of prey also increased, which in this case is algae.
Therefore, we will use a $N(0.5,1)$ prior for $\mathbf{A}$ since the
growth of rotifier population depends on the growth of algae due to in
the predator-prey model the only prey is algae. Now lets determine the
prior distribution for $\mathbf{b}$. Rotifier can split into three
rotifier everyday in a healthy culture, but some times split into two
rotifier can takes around 10 days. This implies sometimes the new
rotifier produces in a single day could only be less than or equal to
three rotifier. As for unicellular algae, some algae can grow fast and
double in just 24 hours due to it belong into micro algae group, an
algae that can double in 24 hours and grow fast. Example of algae that
belong into this group is $\textit{Chlorella}$. By using this
information, it is reasonable to choose
$N\left(\left(\begin{matrix}0\\ 0 \end{matrix}\right), \left(\begin{matrix}1 & 0\\ 0 & 0.6\end{matrix}\right) \right)$
as the prior distribution for $\mathbf{b}$. As for the parameter
$\eta^2_A$, $\eta^2_R$, $\sigma^2_A$, and $\sigma^2_R$ we will choose to
use the same inverse Gamma(0.1, 0.1) prior since the log concentration
of algae should be lower than the log concentration of rotifier because
rotifier reduces the population of algae by consumed them in order to
sustain their life.

Reference regarding rotifier and unicellular algae can be seen in:

- <a
  href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2427.1990.tb00295.x#"
  class="uri">https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2427.1990.tb00295.x</a>

- <https://reefs.com/magazine/the-breeder-s-net-the-rotifer-and-rotifer-home-culture/>

- <https://www.nature.com/articles/s41586-019-1857-0>

- <http://allaboutalgae.com/faq/>

- <https://www.biologyonline.com/dictionary/microalgae>

- [https://medcraveonline.com/MOJFPT/chlorella-and-spirulina-microalgae-as-sources-of-functional-foods-nutraceuticals-and-food-supplements-an-overview.html](https://medcraveonline.com/MOJFPT/chlorella-and-spirulina-microalgae-as-sources-of-functional-foods-nutraceuticals-and-food-supplements-an-overview.html#:~:text=Chlorella%20and%20Spirulina%20are%20two,is%20a%20filamentous%20cyanobacterium%2C%20multicellular.)

Lets implement our new prior in our new JAGS model with 10000 Burn-in
steps.

``` r
#We create the model string in JAGS
model2_string <-   
  "model {
  # prior on the initial true log-concentration at day 0 (x_0), denoted by x[1] in R
  
  x[1,1:2] ~ dmnorm(mu.x0[1:2], prec.x0[1:2, 1:2])
  
  #prior for vector b
  b[1:2] ~ dmnorm(mu.b[1:2], prec.b[1:2, 1:2])
  
  #create matrix A
  for (i in 1:2) {
     for (j in 1:2){
        A[i,j] ~ dnorm(0.5, 1)
     }
  }
  
  #inverse gamma prior for each sigma square and eta square
  tau.A1 ~ dgamma(0.1, 0.1)
  tau.R1 ~ dgamma(0.1, 0.1)
  tau.A2 ~ dgamma(0.1, 0.1)
  tau.R2 ~ dgamma(0.1, 0.1)
  
  sigma2.A <- 1/tau.A1
  sigma2.R <- 1/tau.R1
  eta2.A <- 1/tau.A2
  eta2.R <- 1/tau.R2
  
  #covariance matrix for vector w
  Sigma.w[1,1] <- sigma2.R
  Sigma.w[1,2] <- 0
  Sigma.w[2,1] <- 0
  Sigma.w[2,2] <- sigma2.A
  
  #precision matrix for vector w
  prec.w <- inverse(Sigma.w)
  
  #covariance matrix for vector v
  Sigma.v[1,1] <- eta2.R
  Sigma.v[1,2] <- 0
  Sigma.v[2,1] <- 0
  Sigma.v[2,2] <- eta2.A
  
  #precision matrix for vector v
  prec.v <- inverse(Sigma.v)
  
  #Likelihood
  #n denotes the total number of days considered starting from day 0
  #i = 1 denotes day 0, i = 2 denotes day 1,.....
  for(i in 2:n) {
    #true observed log concentration
    x[i,1:2] ~ dmnorm(A %*% x[i-1,1:2]+b[1:2],prec.w[1:2,1:2])
  }
  
  for(i in 1:n) {
    #observed log concentration
    y1[i] ~ dnorm(x[i,1],prec.v[1,1])
    y2[i] ~ dnorm(x[i,2],prec.v[2,2])
    
    #replicate observed log concentration
    y1rep[i] ~ dnorm(x[i,1],prec.v[1,1])
    y2rep[i] ~ dnorm(x[i,2],prec.v[2,2])
    
  }
  
}"

#specify new hyperparameter for our JAGS model
model2.data <- list(mu.x0 = c(log(6), log(1.5)),mu.b = c(0,0),
                    prec.x0 = solve(diag(4,2)),y1 = y1, n = n,
                    y2 = y2,prec.b = solve(diag(c(1,0.6),2)))

model2 <- jags.model(textConnection(model2_string), 
                     n.chains = 9, data = model2.data)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 735
    ##    Unobserved stochastic nodes: 1154
    ##    Total graph size: 3423
    ## 
    ## Initializing model

``` r
#Burnin for 10000 samples
update(model2,10000,progress.bar="none")

#generate replicate sample for y1 and y2
yrep.res2 <- coda.samples(model2,n.iter=10000,
                          progress.bar="none", 
                          variable.names=c("y1rep","y2rep"))

#store replicate sample for y1 and y2 in a data frame
yrep.rot <- data.frame(rbind(yrep.res2[[1]][,rot_obs],
                             yrep.res2[[2]][,rot_obs],
                             yrep.res2[[3]][,rot_obs],
                             yrep.res2[[4]][,rot_obs],
                             yrep.res2[[5]][,rot_obs],
                             yrep.res2[[6]][,rot_obs],
                             yrep.res2[[7]][,rot_obs],
                             yrep.res2[[8]][,rot_obs],
                             yrep.res2[[9]][,rot_obs]))

yrep.alg <- data.frame(rbind(yrep.res2[[1]][,alg_obs],
                             yrep.res2[[2]][,alg_obs],
                             yrep.res2[[3]][,alg_obs],
                             yrep.res2[[4]][,alg_obs],
                             yrep.res2[[5]][,alg_obs],
                             yrep.res2[[6]][,alg_obs],
                             yrep.res2[[7]][,alg_obs],
                             yrep.res2[[8]][,alg_obs],
                             yrep.res2[[9]][,alg_obs]))

#compute mean, minimum, and maximum value of replicate y1
yrep.rot.min <- apply(yrep.rot,MARGIN=1, FUN=min)
yrep.rot.max <- apply(yrep.rot,MARGIN=1, FUN=max)
yrep.rot.median <- apply(yrep.rot,MARGIN=1, FUN=median)

#compute mean, minimum, and maximum value of replicate y2
yrep.alg.min <- apply(yrep.alg,MARGIN=1, FUN=min)
yrep.alg.max <- apply(yrep.alg,MARGIN=1, FUN=max)
yrep.alg.median <- apply(yrep.alg,MARGIN=1, FUN=median)

#plot the predictive distribution histogram
par(mfrow=c(3,2))
hist(yrep.rot.min,col="gray40", cex.main = 0.75, 
     xlab ='min rotifier',
     main="Rotifier Predictive Distribution for Min")
abline(v=min(y1, na.rm = T),col="red",lwd=2)

hist(yrep.alg.min,col="gray40", cex.main = 0.75,
     xlab = 'min algae',
     main="Algae Predictive Distribution for Min")
abline(v=min(y2, na.rm = T),col="red",lwd=2)

hist(yrep.rot.max,col="gray40", cex.main = 0.75,
     xlab = 'max rotifier',
     main="Rotifier Predictive Distribution for Max")
abline(v=max(y1, na.rm = T),col="red",lwd=2)

hist(yrep.alg.max,col="gray40", cex.main = 0.75,
     xlab = 'min algae',
     main="Algae Predictive Distribution for Max")
abline(v=max(y2, na.rm = T),col="red",lwd=2)

hist(yrep.rot.median,col="gray40", cex.main = 0.75,
     xlab='median rotifier',
     main="Rotifier Predictive Distribution for Median")
abline(v=median(y1, na.rm = T),col="red",lwd=2)


hist(yrep.alg.median,col="gray40", cex.main = 0.75,
     xlab = 'median algae',
     main="Algae Predictive Distribution for Median")
abline(v=median(y2, na.rm = T),col="red",lwd=2)
```

![](assignment-1_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The histograms above show posterior predictive distribution for the
minimum (top), maximum (middle) and median(bottom) log concentration of
rotifier (all plot in the left) and algae (all plot in the right) for
the Bayesian fit to the concentration of rotifier and algae time series
data. The descriptive statistic values (minimum, maximum and median) in
the observed data set for the log concentration of rotifier and algae
are shown by vertical red lines. The lines seem to be within the typical
range of the replicates, showing a reasonably good fit. We can see that
the shape of predictive distribution for the descriptive statistic
values (minimum, maximum and median) of the replicates of log
concentration of rotifier all shaped like a normal distribution curve,
while for the replicates of log concentration of algae only the
histogram related to the predictive distribution of the median that
shaped like a normal distribution. The predictive distribution of the
replicates of log concentration of algae for the minimum and maximum
values are shaped like a left-skewed an right-skewed distribution curve
respectively.

The DIC for our new JAGS model is presented below. Comparing to the DIC
of our first JAGS model in part c), we can see that the alternative
prior that we have chosen for our JAGS model based on external
information results in a significantly smaller DIC value, indicating
better fit on the data.

``` r
#DIC score for our new JAGS model
dic.samples(model2, n.iter=10000)
```

    ## Mean deviance:  -174.3 
    ## penalty 599.5 
    ## Penalized deviance: 425.2

**e)\[10 marks\] Update the model with your informative prior in part d)
to compute the posterior distribution of the log concentrations sizes
(**$\mathbf{x}_t$**) on the days 376-395 (20 additional days).**

**Plot the evolution of the posterior mean of the log concentrations for
rotifier and algae during days 376-395 on a single plot, along with
curves that correspond to the \[2.5%, 97.5%\] credible interval of the
log concentration size (**$\mathbf{x}_t$) according to the posterior
distribution at each year \[Hint: you need $2+2\cdot 2 = 6$ curves in
total, use different colours for the curves for rotifier and algae\].

**Finally, estimate the posterior probability that the concentration of
algae (measured in 10^6 algae/ml, as in the data) becomes smaller than**
$0.1$ **at any time during this 20 additional days (days 376-395).**

According to the plot below, we can see that the trend of the change of
rotifier and algae log concentration during this 20 additional days
(days 376-395) is showing a fluctuation since the log concentration of
rotifier and algae experienced a decreased at some period then also
experienced an increasing in the next period or in the previous period.
In addition, we also can see that the posterior mean of the log
concentration of rotifier and algae lies inside their respective 95%
credible interval of the log concentration size ($x_t$).

Now we want to estimate the posterior probability that the concentration
of algae (measured in $10^6$ algae/ml, as in the data) becomes smaller
at any time during this 20 additional days (days 376-395), which is
$\text{P}\left(\min_{t \in\{376,377,\dots,395\}}x_t < \text{log}(0.1)\right)$.
According to the output below, we can see that the posterior probability
that the concentration of algae becomes smaller at any time during days
376-395 is 0.0144.

``` r
#extend data until day 395
y1_new <- c(y1,rep(NA,20))
y2_new <- c(y2,rep(NA,20))

#specify new hyperparameter for our JAGS model
model2.data <- list(mu.x0 = c(log(6), log(1.5)),mu.b = c(0,0),
                    prec.x0 = solve(diag(4,2)),y1 = y1_new,n =396,
                    y2 = y2_new,prec.b = solve(diag(c(1,0.6),2)))

#parameter to be sample
sample_param <- c("x[377,1]","x[378,1]", "x[379,1]","x[380,1]",
                  "x[381,1]","x[382,1]","x[383,1]","x[384,1]",
                  "x[385,1]","x[386,1]","x[387,1]","x[388,1]",
                  "x[389,1]","x[390,1]","x[391,1]","x[392,1]",
                  "x[393,1]","x[394,1]","x[395,1]","x[396,1]",
                  "x[377,2]","x[378,2]","x[379,2]","x[380,2]",
                  "x[381,2]","x[382,2]","x[383,2]","x[384,2]",
                  "x[385,2]","x[386,2]","x[387,2]","x[388,2]",
                  "x[389,2]","x[390,2]","x[391,2]","x[392,2]",
                  "x[393,2]","x[394,2]","x[395,2]","x[396,2]")

#input the new hyperparameter to our JAGS model
model2 <- jags.model(textConnection(model2_string),n.chains = 9,
                     data = model2.data)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 735
    ##    Unobserved stochastic nodes: 1254
    ##    Total graph size: 3603
    ## 
    ## Initializing model

``` r
#Burnin for 10000 samples
update(model2,10000,progress.bar="none")

# Running the model
res.model2 <- coda.samples(model2,variable.names=sample_param, 
                           n.iter=10000,progress.bar="none")

#We combine the results from all chains into a single dataframe
xres.rot <- data.frame(rbind(res.model2[[1]][,1:20],
                             res.model2[[2]][,1:20],
                             res.model2[[3]][,1:20],
                             res.model2[[4]][,1:20],
                             res.model2[[5]][,1:20],
                             res.model2[[6]][,1:20],
                             res.model2[[7]][,1:20],
                             res.model2[[8]][,1:20],
                             res.model2[[9]][,1:20]))

xres.alg <- data.frame(rbind(res.model2[[1]][,21:40],
                             res.model2[[2]][,21:40],
                             res.model2[[3]][,21:40],
                             res.model2[[4]][,21:40],
                             res.model2[[5]][,21:40],
                             res.model2[[6]][,21:40],
                             res.model2[[7]][,21:40],
                             res.model2[[8]][,21:40],
                             res.model2[[9]][,21:40]))

#compute posterior mean and 95% credible interval for rotifier

xrot.mean <- apply(xres.rot,MARGIN=2, FUN=mean)
xrot.q025 <- apply(xres.rot, MARGIN=2,
                   FUN=function(x) quantile(x,prob=0.025))
xrot.q975 <- apply(xres.rot, MARGIN=2,
                   FUN=function(x) quantile(x,prob=0.975))

#compute posterior mean and 95% credible interval for algae
xalg.mean <- apply(xres.alg,MARGIN=2, FUN=mean)
xalg.q025 <- apply(xres.alg, MARGIN=2,
                   FUN=function(x) quantile(x,prob=0.025))
xalg.q975 <- apply(xres.alg, MARGIN=2,
                   FUN=function(x) quantile(x,prob=0.975))

#plot the posterior mean and 95% confidence interval for rotifier
upper_lim <- max(c(xrot.mean,xrot.q025,xrot.q975,xalg.mean,
                    xalg.q025,xalg.q975),na.rm=T)

lower_lim <- min(c(xrot.mean,xrot.q025,xrot.q975,xalg.mean,
                    xalg.q025,xalg.q975),na.rm=T)

plot(376:395, xrot.mean,type="l",col = 'red',cex.main = 0.75, 
     cex.lab = 0.75,xlab="Days",ylab="Log concentration",
     ylim = c(lower_lim, upper_lim+3),
     main="Posterior Mean and 95% Credible Interval for Log Concentration")

lines(376:395, xrot.q025,lty=3,col="dark red")
lines(376:395, xrot.q975,lty=3,col="dark red")

#plot the posterior mean and 95%confidence interval for algae
lines(376:395, xalg.mean,col = 'blue')
lines(376:395, xalg.q025,lty=3,col="deep sky blue")
lines(376:395, xalg.q975,lty=3,col="deep sky blue")

#add legend to the plot
plot_title <- c("Posterior Mean (Rotifier)","Posterior Mean (Algae)",
                "95% CI (Rotifier)", "95% CI (Algae)")

legend("topright", col=c("red","blue","dark red","deep sky blue"),
       lty=c(1,1,3,3), cex=0.75, bty = 'n',legend=plot_title)
```

![](assignment-1_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
#posterior probability that algae concentration < 0.1 in 20 new day
xres.alg.min <- apply(xres.alg,MARGIN=1, FUN=min)
p <- mean(xres.alg.min<log(0.1),na.rm=T)
cat("Posterior Probability algae concentration < 0.1:",p,"\n")
```

    ## Posterior Probability algae concentration < 0.1: 0.01521111

<img src="horse_racing.jpg" style="width:100.0%" />

**Problem 2 - Horse racing data**

**In this problem, we are going to construct a predictive model for
horse races. The dataset (races.csv and runs.csv) contains the
information about 1000 horse races in Hong Kong during the years
1997-1998 (originally from <https://www.kaggle.com/gdaley/hkracing>).
Races.csv contains information about each race (such as distance, venue,
track conditions, etc.), while runs.csv contains information about each
horse participating in each race (such as finish time in the race).
Detailed description of all columns in these files is available in the
file horse_racing_data_info.txt.**

**Our goal is to model the mean speed of each horse during the races
based on covariates available before the race begins.**

**We are going to use INLA to fit several different regression models to
this dataset. First, we load ILNA and the datasets and display the first
few rows.**

``` r
library(INLA)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loading required package: parallel

    ## Loading required package: sp

    ## This is INLA_22.05.07 built 2022-05-07 09:52:03 UTC.
    ##  - See www.r-inla.org/contact-us for how to get help.

``` r
#If it loaded correctly, you should see this in the output:
#Loading required package: Matrix
#Loading required package: foreach
#Loading required package: parallel
#Loading required package: sp
#This is INLA_21.11.22 built 2021-11-21 16:13:28 UTC.
# - See www.r-inla.org/contact-us for how to get help.
# - To enable PARDISO sparse library; see inla.pardiso()

#The following code does the full installation. You can try it if INLA has not been installed.
#First installing some of the dependencies
#install.packages("BiocManager")
#BiocManager::install("Rgraphviz")
#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install("graph")
#Installing INLA
#install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
#library(INLA)
```

``` r
runs <- read.csv(file = 'runs.csv')
head(runs)
```

    ##   race_id horse_no horse_id result won lengths_behind horse_age horse_country
    ## 1       0        1     3917     10   0           8.00         3           AUS
    ## 2       0        2     2157      8   0           5.75         3            NZ
    ## 3       0        3      858      7   0           4.75         3            NZ
    ## 4       0        4     1853      9   0           6.25         3           SAF
    ## 5       0        5     2796      6   0           3.75         3            GB
    ## 6       0        6     3296      3   0           1.25         3            NZ
    ##   horse_type horse_rating horse_gear declared_weight actual_weight draw
    ## 1    Gelding           60         --            1020           133    7
    ## 2    Gelding           60         --             980           133   12
    ## 3    Gelding           60         --            1082           132    8
    ## 4    Gelding           60         --            1118           127   13
    ## 5    Gelding           60         --             972           131   14
    ## 6    Gelding           60         --            1114           127    5
    ##   position_sec1 position_sec2 position_sec3 position_sec4 position_sec5
    ## 1             6             4             6            10            NA
    ## 2            12            13            13             8            NA
    ## 3             3             2             2             7            NA
    ## 4             8             8            11             9            NA
    ## 5            13            12            12             6            NA
    ## 6            11            11             5             3            NA
    ##   position_sec6 behind_sec1 behind_sec2 behind_sec3 behind_sec4 behind_sec5
    ## 1            NA        2.00        2.00        1.50        8.00          NA
    ## 2            NA        6.50        9.00        5.00        5.75          NA
    ## 3            NA        1.00        1.00        0.75        4.75          NA
    ## 4            NA        3.50        5.00        3.50        6.25          NA
    ## 5            NA        7.75        8.75        4.25        3.75          NA
    ## 6            NA        5.00        7.75        1.25        1.25          NA
    ##   behind_sec6 time1 time2 time3 time4 time5 time6 finish_time win_odds
    ## 1          NA 13.85 21.59 23.86 24.62    NA    NA       83.92      9.7
    ## 2          NA 14.57 21.99 23.30 23.70    NA    NA       83.56     16.0
    ## 3          NA 13.69 21.59 23.90 24.22    NA    NA       83.40      3.5
    ## 4          NA 14.09 21.83 23.70 24.00    NA    NA       83.62     39.0
    ## 5          NA 14.77 21.75 23.22 23.50    NA    NA       83.24     50.0
    ## 6          NA 14.33 22.03 22.90 23.57    NA    NA       82.83      7.0
    ##   place_odds trainer_id jockey_id
    ## 1        3.7        118         2
    ## 2        4.9        164        57
    ## 3        1.5        137        18
    ## 4       11.0         80        59
    ## 5       14.0          9       154
    ## 6        1.8         54        34

``` r
races<- read.csv(file = 'races.csv')
head(races)
```

    ##   race_id       date venue race_no config surface distance        going
    ## 1       0 1997-06-02    ST       1      A       0     1400 GOOD TO FIRM
    ## 2       1 1997-06-02    ST       2      A       0     1200 GOOD TO FIRM
    ## 3       2 1997-06-02    ST       3      A       0     1400 GOOD TO FIRM
    ## 4       3 1997-06-02    ST       4      A       0     1200 GOOD TO FIRM
    ## 5       4 1997-06-02    ST       5      A       0     1600 GOOD TO FIRM
    ## 6       5 1997-06-02    ST       6      A       0     1200 GOOD TO FIRM
    ##   horse_ratings   prize race_class sec_time1 sec_time2 sec_time3 sec_time4
    ## 1         40-15  485000          5     13.53     21.59     23.94     23.58
    ## 2         40-15  485000          5     24.05     22.64     23.70        NA
    ## 3         60-40  625000          4     13.77     22.22     24.88     22.82
    ## 4        120-95 1750000          1     24.33     22.47     22.09        NA
    ## 5         60-40  625000          4     25.45     23.52     23.31     23.56
    ## 6         60-40  625000          4     23.47     22.48     23.25        NA
    ##   sec_time5 sec_time6 sec_time7 time1 time2 time3 time4 time5 time6 time7
    ## 1        NA        NA        NA 13.53 35.12 59.06 82.64    NA    NA    NA
    ## 2        NA        NA        NA 24.05 46.69 70.39    NA    NA    NA    NA
    ## 3        NA        NA        NA 13.77 35.99 60.87 83.69    NA    NA    NA
    ## 4        NA        NA        NA 24.33 46.80 68.89    NA    NA    NA    NA
    ## 5        NA        NA        NA 25.45 48.97 72.28 95.84    NA    NA    NA
    ## 6        NA        NA        NA 23.47 45.95 69.20    NA    NA    NA    NA
    ##   place_combination1 place_combination2 place_combination3 place_combination4
    ## 1                  8                 11                  6                 NA
    ## 2                  5                 13                  4                 NA
    ## 3                 11                  1                 13                 NA
    ## 4                  5                  3                 10                 NA
    ## 5                  2                 10                  1                 NA
    ## 6                  9                 14                  8                 NA
    ##   place_dividend1 place_dividend2 place_dividend3 place_dividend4
    ## 1            36.5            25.5            18.0              NA
    ## 2            12.5            47.0            33.5              NA
    ## 3            23.0            23.0            59.5              NA
    ## 4            14.0            24.5            16.0              NA
    ## 5            15.5            28.0            17.5              NA
    ## 6            16.5           408.0            70.0              NA
    ##   win_combination1 win_dividend1 win_combination2 win_dividend2
    ## 1                8         121.0               NA            NA
    ## 2                5          23.5               NA            NA
    ## 3               11          70.0               NA            NA
    ## 4                5          52.0               NA            NA
    ## 5                2          36.5               NA            NA
    ## 6                9          61.0               NA            NA

**a)\[10 marks\] Create a dataframe that includes the mean speed of each
horse in each race and the distance of the race in a column \[Hint: you
can do this adding two extra columns to the runs dataframe\].**

**Fit a linear regression model (lm) with the mean speed as a response
variable. The covariates should be the horse id as a categorical
variable, and the race distance, horse rating, and horse age as standard
variable. Scale the non-categorical covariates before fitting the model
(i.e. center and divide by their standard deviation, you can use the**
`scale` **function in R for this).**

**Print out the summary of the lm model, discuss the quality of the
fit.**

From the summary of our linear regression model, we can see that the
P-values for the coefficient of intercept and race distance is less than
0.05, which implies that the the intercept and race distance give
significant effect to our response variable (mean speed of each horse).
However, the P-values for the coefficient of horse rating, horse age,
and horse id are greater than 0.05 even though there is a P-value for
the coefficient of horse id that is less than 0.05. This implies that
the horse rating, horse age, and horse id did not give significant
effect to our response variable (mean speed of each horse). The
estimated coefficient of the intercept is 16.69, which implies that the
average mean speed of each horse will be equal to 16.69 if all observed
values of the explanatory variable (horse age, horse rating, race
distance, and horse id dummy variable) in our model equals to zero. Now
lets evaluate our linear regression model based on the residual standard
error and multiple R-squared. We can see that the residual standard
error is 0.2038, which implies that the linear regression model predicts
the mean speed of each horse with an average error around 20.38%.
Furthermore, according to our multiple R-squared value which equal to
0.7883, we can said that our regression model fits 78.83% of the mean
speed data by just including horse id, horse age, horse rating, and race
distance as the explanatory variable in our model.

``` r
options(max.print=100)
#get the distance and mean of speed
runs_data <- merge(runs, races[,c('race_id','distance')],
                   by = 'race_id')

runs_data['speed'] <- runs_data$distance/runs_data$finish_time

#convert horse id data to categorical variable
runs_data$horse_id <- as.factor(runs_data$horse_id)

#scale all numerical variable covariates that include in the model 
runs_data$distance <- scale(runs_data$distance)
runs_data$horse_rating <- scale(runs_data$horse_rating)
runs_data$horse_age <- scale(runs_data$horse_age)

#fit linear model and display the model summary
fit_lm <- lm(speed~distance+horse_rating+horse_age+horse_id,
             data = runs_data)
summary(fit_lm)
```

    ## 
    ## Call:
    ## lm(formula = speed ~ distance + horse_rating + horse_age + horse_id, 
    ##     data = runs_data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2.64378 -0.09259  0.01438  0.11849  0.59029 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error  t value Pr(>|t|)    
    ## (Intercept)   1.669e+01  4.461e-02  374.039  < 2e-16 ***
    ## distance     -3.524e-01  3.196e-03 -110.264  < 2e-16 ***
    ## horse_rating  5.166e-04  3.546e-03    0.146 0.884185    
    ## horse_age     5.659e-03  3.222e-03    1.756 0.079078 .  
    ## horse_id29   -1.449e-01  7.197e-02   -2.013 0.044162 *  
    ## horse_id61   -2.604e-01  1.114e-01   -2.338 0.019404 *  
    ## horse_id62   -9.859e-02  1.016e-01   -0.970 0.331919    
    ## horse_id63   -8.021e-02  8.924e-02   -0.899 0.368752    
    ## horse_id64    6.383e-02  1.510e-01    0.423 0.672519    
    ## horse_id65   -1.070e-01  1.015e-01   -1.054 0.292048    
    ## horse_id66   -6.092e-02  8.905e-02   -0.684 0.493931    
    ## horse_id67   -7.032e-02  8.490e-02   -0.828 0.407527    
    ## horse_id69   -1.243e+00  2.087e-01   -5.956 2.67e-09 ***
    ## horse_id70    1.308e-01  1.041e-01    1.257 0.208848    
    ## horse_id72   -8.189e-03  7.200e-02   -0.114 0.909452    
    ## horse_id79   -2.675e-01  1.258e-01   -2.127 0.033463 *  
    ## horse_id80    3.551e-02  7.831e-02    0.453 0.650228    
    ## horse_id82    1.316e-01  6.547e-02    2.010 0.044455 *  
    ## horse_id91   -2.694e-02  1.016e-01   -0.265 0.790933    
    ## horse_id92    2.178e-01  1.016e-01    2.144 0.032079 *  
    ##  [ reached getOption("max.print") -- omitted 1415 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.2038 on 11109 degrees of freedom
    ## Multiple R-squared:  0.7883, Adjusted R-squared:  0.761 
    ## F-statistic: 28.85 on 1434 and 11109 DF,  p-value: < 2.2e-16

``` r
#plot fitted values vs residuals
plot(fit_lm$fitted.values, fit_lm$residuals, xlab = 'Fitted values', 
     ylab = 'Resiudals', main = 'Residuals vs Fitted')
abline(h = 0, lty = 3, col = 'black')
lines(lowess(fit_lm$fitted.values, fit_lm$residuals), col = 'red')
```

![](assignment-1_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

According to the residual vs fitted values plot, we can see that the
residuals are roughly form a “horizontal band” around the horizontal
zero line. This indicates that the variances of residuals are equal
which implies that there is no heteroscedasticity in our model. We also
can observe that there exist two outlier point from the residuals vs
fitted values plot since that two points distance is very far from other
points that mostly gather around the zero horizontal line. Moreover, we
also can see that the linear relationship between the target (mean
speed) and its predictor variable(horse id, horse age, horse rating, and
race distance) is clearly captured since the red line in the residuals
vs fitted values plot did not show an indication of non-linear line in
the plot.

**b)\[10 marks\] Fit the same model in INLA (i.e. Bayesian linear
regression with Gaussian likelihood, mean speed is the response
variable, and the same covariates used with scaling for the
non-categorical covariates). Set a Gamma (0.1,0.1) prior for the
precision, and Gaussian priors with mean zero and variance 1000000 for
all of the regression coefficients (including the intercept).**

**Print out the summary of the INLA model. Compute the posterior mean of
the variance parameter** $\sigma^2$**. Plot the posterior density for
the variance parameter** $\sigma^2$**. Compute the negative sum log CPO
(NSLCPO) and DIC values for this model (smaller values indicate better
fit).**

**Compute the standard deviation of the mean residuals (i.e. the
differences between the posterior mean of the fitted values and the true
response variable).**

**Discuss the results.**

First, we will fit Bayesian linear regression with Gaussian likelihood
model using INLA and display its model summary. We will use the same
response and explanatory variable just like the variable we use in a).
According to our model summary below, we can see that the posterior mean
for our regression model parameter are not significantly different from
the estimated values of our linear regression parameter in a). The
posterior mean of the intercept parameter is 16.687, which implies that
the average mean speed of each horse will be equal to 16.687 if all
observed values of the explanatory variable (horse age, horse rating,
race distance, and horse id dummy variable) in our model equals to zero.
As we can see from the model summary below, the column
$\verb|0.025quant|$ and $\verb|0.975quant|$ represents the lower and
upper bound of a 95% credible interval for our regression parameter. In
Bayesian statistics, we can interpret the 95% credible interval for the
regression parameter as a fix interval where the value of interest of
our regression parameter will lies on that interval with a 95%
probability.

``` r
options(max.print=100)
#prior for beta and precision
prec.prior <- list(prec=list(prior="loggamma",param=c(0.1,0.1)))
prior.beta <- list(mean.intercept = 0,prec.intercept = 1e-06,
                   mean = 0, prec = 1e-06)

#input the regression model to INLA 
m.I <- inla(speed~distance+horse_rating+horse_age+horse_id,
            data=runs_data, family="gaussian",
            control.predictor = list(compute = T),
            control.compute = list(cpo=T,dic = T,config = T),
            control.family=list(hyper=prec.prior),
            control.fixed=prior.beta)

#display model summary
summary(m.I)
```

    ## 
    ## Call:
    ##    c("inla.core(formula = formula, family = family, contrasts = contrasts, 
    ##    ", " data = data, quantiles = quantiles, E = E, offset = offset, ", " 
    ##    scale = scale, weights = weights, Ntrials = Ntrials, strata = strata, 
    ##    ", " lp.scale = lp.scale, link.covariates = link.covariates, verbose = 
    ##    verbose, ", " lincomb = lincomb, selection = selection, control.compute 
    ##    = control.compute, ", " control.predictor = control.predictor, 
    ##    control.family = control.family, ", " control.inla = control.inla, 
    ##    control.fixed = control.fixed, ", " control.mode = control.mode, 
    ##    control.expert = control.expert, ", " control.hazard = control.hazard, 
    ##    control.lincomb = control.lincomb, ", " control.update = 
    ##    control.update, control.lp.scale = control.lp.scale, ", " 
    ##    control.pardiso = control.pardiso, only.hyperparam = only.hyperparam, 
    ##    ", " inla.call = inla.call, inla.arg = inla.arg, num.threads = 
    ##    num.threads, ", " blas.num.threads = blas.num.threads, keep = keep, 
    ##    working.directory = working.directory, ", " silent = silent, inla.mode 
    ##    = inla.mode, safe = FALSE, debug = debug, ", " .parent.frame = 
    ##    .parent.frame)") 
    ## Time used:
    ##     Pre = 7.94, Running = 23.7, Post = 4314, Total = 4346 
    ## Fixed effects:
    ##                mean    sd 0.025quant 0.5quant 0.975quant mode kld
    ## (Intercept)  16.687 0.045     16.599   16.687     16.774   NA   0
    ## distance     -0.352 0.003     -0.359   -0.352     -0.346   NA   0
    ## horse_rating  0.001 0.004     -0.006    0.001      0.007   NA   0
    ## horse_age     0.006 0.003     -0.001    0.006      0.012   NA   0
    ## horse_id29   -0.145 0.072     -0.286   -0.145     -0.004   NA   0
    ## horse_id61   -0.260 0.111     -0.479   -0.260     -0.042   NA   0
    ## horse_id62   -0.099 0.102     -0.298   -0.099      0.101   NA   0
    ## horse_id63   -0.080 0.089     -0.255   -0.080      0.095   NA   0
    ## horse_id64    0.064 0.151     -0.232    0.064      0.360   NA   0
    ## horse_id65   -0.107 0.102     -0.306   -0.107      0.092   NA   0
    ## horse_id66   -0.061 0.089     -0.236   -0.061      0.114   NA   0
    ## horse_id67   -0.070 0.085     -0.237   -0.070      0.096   NA   0
    ## horse_id69   -1.243 0.209     -1.652   -1.243     -0.834   NA   0
    ## horse_id70    0.131 0.104     -0.073    0.131      0.335   NA   0
    ##  [ reached getOption("max.print") -- omitted 1421 rows ]
    ## 
    ## Model hyperparameters:
    ##                                          mean    sd 0.025quant 0.5quant
    ## Precision for the Gaussian observations 24.06 0.323      23.44    24.06
    ##                                         0.975quant mode
    ## Precision for the Gaussian observations      24.70   NA
    ## 
    ## Deviance Information Criterion (DIC) ...............: -3391.40
    ## Deviance Information Criterion (DIC, saturated) ....: 414792.19
    ## Effective number of parameters .....................: 1174.34
    ## 
    ## Marginal log-Likelihood:  -10677.48 
    ## CPO, PIT is computed 
    ## Posterior summaries for the linear predictor and the fitted values are computed
    ## (Posterior marginals needs also 'control.compute=list(return.marginals.predictor=TRUE)')

Observe that the values of the posterior mean and standard deviation for
the posterior distribution of the horse rating, horse age, and some of
horse id coefficient are close to 0. This implies that the regression
parameter is likely not important in the model. However, I might still
reconsider to include the variable horse age in the linear regression
model since age is confounding variable. As we know, confounding
variable is a variable that influences the response variable (younger
horse can run much more faster than older horse) and also the predictor
variable (e.g. a horse with higher speed can run to reach a very far
distance than a horse with lower speed).

In our Bayesian linear regression model summary, we are given the
descriptive statistics for the precision $\tau$. This parameter has
posterior mean and standard deviation equals to 24.06 and 0.318
respectively. In addition, we also obtain the 95% credible interval for
the precision $\tau$, which is (23.44, 24.69). Therefore, we can say
that the actual values of the precision $\tau$ will lies between 23.44
and 24.69 with probability 95%. Since in our Bayesian linear regression
model we are given given the marginal of the precision $\tau$, then we
need to obtain the marginal of the variance $\sigma^2$ by using the
$\verb|inla.tmargimal|$ function in $\verb|R|$.

Observe that in $\verb|inla.tmarginal|$ function we defined a function
that contains a formula to compute the variance $\sigma^2$ by using the
precision $\tau$. As we know, the formula that describe the relationship
between variance $\sigma^2$ and precision $\tau$ is
$\sigma^2 = \frac{1}{\tau}$. After we obtain the marginal of the
variance $\sigma^2$, we can plot the posterior density of $\sigma^2$.
According to the posterior density curve for the variance $\sigma^2$, we
can see that the posterior distribution for $\sigma^2$ is a normal
distribution. In addition, we also obtain a summary statistics about the
posterior mean and standard deviation for $\sigma^2$, which is 0.0415648
and 0.00055411 respectively. Not only that, we also obtain the 2.5% and
97.5% quantile for $\sigma^2$, which is 0.0404908 and 0.0426646
respectively. As we already know, the 2.5% and 97.5% quantile for
$\sigma^2$ represents the lower and upper bound for the 95% credible
interval for the variance $\sigma^2$. Therefore, we can said that the
true values of the variance $\sigma^2$ will lies between 0.0415648 and
0.00055411 with a probability 95%.

``` r
#Summary statistics of sigma2
marg.sigma2 <- inla.tmarginal(function(tau) tau^(-1),
                              m.I$marginals.hyperpar[[1]])
cat("Summary statistics of sigma square: \n")
```

    ## Summary statistics of sigma square:

``` r
inla.zmarginal(marg.sigma2)
```

    ## Mean            0.0415633 
    ## Stdev           0.000549977 
    ## Quantile  0.025 0.0404952 
    ## Quantile  0.25  0.0411849 
    ## Quantile  0.5   0.0415574 
    ## Quantile  0.75  0.041935 
    ## Quantile  0.975 0.0426572

``` r
#Plot of marginal of sigma
plot(marg.sigma2, type ="l",xlab=expression(sigma^2),
     ylab="Density", 
     main='Posterior density of'~sigma^2~'for model 1')
```

![](assignment-1_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
#compute NSLCPO
cat("NSLCPO:",-sum(log(m.I$cpo$cpo)),"\n")
```

    ## NSLCPO: -1223.601

``` r
#compute DIC
cat("DIC:",m.I$dic$dic,"\n")
```

    ## DIC: -3391.403

``` r
#compute standard deviation of mean residuals
sdmr <- sd(runs_data$speed-m.I$summary.fitted.values$mean)
cat("Standard deviation of mean residuals for model 1:",sdmr,"\n")
```

    ## Standard deviation of mean residuals for model 1: 0.1918068

According to the model summary above, we can see that the NSLCPO and DIC
are -1231.658 and -2941.842 respectively. Thus we can said that our
NSLCPO and DIC are quite small. This implies that our model fits well on
the data. As we can observe here, the standard deviation of the mean
residuals is 0.1918068 , which is slightly different from the residual
standard error of the linear regression model in a).

**c)\[10 marks\] In this question, we are going to improve the model in
b) by using more informative priors and more columns from the dataset.**

**First, using some publicly available information from the internet
(Hint: use Google search) find out about the typical speed of race
horses in Hong Kong, and use this information to construct a prior for
the intercept. Explain the rationale for your choice.**

**Second, look through all of the information in the datasets that is
available before the race (Hint: you need to read the description
horse_racing_data_info.txt for information about the columns. position,
behind, result, won, and time related columns are not available before
the race). Discuss your rationale for including some of these in the
dataset (make sure to scale them if they are non-categorical).**

**Feel free to try creating additional covariates such as polynomial or
interaction terms (Hint: this can be done using I() in the formula), and
you can also try to use a different likelihood (such as Student-t
distribution).**

**Fit your new model in INLA (i.e. Bayesian linear regression, mean
speed is the response variable, and scaling done for the non-categorical
covariates).**

**Print out the summary of the INLA model. Compute the negative sum log
CPO (NSLCPO) and DIC values for this model (smaller values indicate
better fit).**

**Compute the standard deviation of the mean residuals (i.e. the
differences between the posterior mean of the fitted values and the true
response variable).**

**Discuss the results and compare your model to the model from b).**

**Please only include your best performing model in the report.**

All the race horses in Hong Kong are thoroughbreds. This horse are
considered hot-blooded horse that are known for their agility, speed,
and spirit. It has average running speed around 35 to 44 miles per
hours, which is equivalent to around 15.65 to 19.67 meters per seconds.
Overall, race horses speeds is approximately 40 to 44 miles per hours
that can be reach for less than 20 seconds. However, most of them can’t
run faster than 20 to 30 miles per hour with a raider in their back. As
we know, the top speed of the fastest horse is 55 miles per hours or
equivalent to around 24.59 meters per second. American Quarter Horses
can reach it while sprinting a distance shorter than 400 m. The
reference about race horses and its speed that we used in this report
are provided as follows:

- <https://entertainment.hkjc.com/entertainment/english/learn-racing/racing-legacy-and-encyclopedia/know-about-horses.aspx>

- <a
  href="https://www.deephollowranch.com/horse-speed/#:~:text=The%20average%20racehorses%20speed%20is,for%20less%20than%2020%20seconds."
  class="uri">&lt;https://www.deephollowranch.com/horse-speed/&gt;</a>

- [\<https://horseracingsense.com/speed-worlds-fastest-horses/\>](https://horseracingsense.com/speed-worlds-fastest-horses/#:~:text=55%20MPH%20is%20the%20top%20speed%20of%20the%20world's%20fastest%20horses.)

- [https://en.wikipedia.org/wiki/Thoroughbred](https://en.wikipedia.org/wiki/Thoroughbred#:~:text=The%20Thoroughbred%20is%20a%20horse,agility%2C%20speed%2C%20and%20spirit.)

Now we will construct the new normal distribution prior for the
intercept $\beta_0$ using the information about race horse speed in the
reference above. Let $\mu_{\beta_0}$, $\sigma^2_{\beta_0}$, and
$\tau_{\beta_0}$ be the mean, variance, and precision parameter for the
normal distribution prior for the intercept $\beta_0$. Let $\text{P}(A)$
be the probability that event $A$ will occur. Since the average running
speed of thoroughbreds is between 15.65 and 19.67 meters per seconds,
then it is reasonable to choose $\mu_{\beta_0}$ around 17.66 seems
reasonable. Recall that the intercept $\beta_0$ is the mean speed when
the observed values of all covariates are equal to zero. Therefore, the
intercept $\beta_0$ must be not greater than 24.59 meters per second.
This implies that $\text{P}(\beta_0 > 24.59)$ = 0. Thus,
$\text{P}(\beta_0 \leq 24.59)$ = 1. Since in the normal distribution
table the highest probability is $\text{P}(Z \leq 3.49)$ = 0.9998 where
$Z$ is a standard normal random variable, then we will use the
approximation
$\text{P}(\beta_0 \leq 24.59) \approx \text{P}\left(Z \leq 3.49\right)$.
Therefore, we will have
$$\text{P}(\beta_0 \leq 24.59) = \text{P}\left(Z \leq \frac{24.59-\mu_{\beta_0}}{\sigma_{\beta_0}}\right) = \text{P}\left(Z \leq \frac{24.59-17.66}{\sigma_{\beta_0}}\right)  = \text{P}\left(Z \leq \frac{6.93}{\sigma_{\beta_0}}\right)\approx \text{P}(Z \leq 3.49).$$Thus,
we will obtain $$\frac{6.93}{\sigma_{\beta_0}} \approx 3.49,$$which
implies that the standard deviation $\sigma_{\beta_0}\approx 1.985673$.
In other words, we can obtain the standard deviation
$\sigma^2_{\beta_0} \approx 3.942897$ and the precision
$\tau_{\beta_0} = \frac{1}{\sigma^2_{\beta_0}} \approx 0.2536206$. By
using this results, we obtain the prior distribution for the intercept
$\beta_0$ is $N(17.66,3.942897)$.

Before we use the new prior for the intercept $\beta_0$ for our Bayesian
linear regression model, we will choose the explanatory variable that we
are going to use in this model. For choosing the predictor variable that
we are going to use, we will use the reference about race horse and some
information in HKJC website
([\[https://racing.hkjc.com/racing/\](https://racing.hkjc.com/racing/english)](https://racing.hkjc.com/racing/english/racing-info/racing_course.aspx))
. Usually the speed of a horse is affected by breed, health, age,
airflow through their respiratory system, individual characteristics,
and the weight the horses carry during the race. If we looking at the
runs.csv data, we have the data related to the horse age and the
declared weight (weight of the horse plus weight of the jockey in lbs)
that the horse carried. This two variable clearly have effect to the
horse speed because if the horse become older then its running speed
will decrease and also if the horse carried too much weight then they
will walk much more slower than its original speed.

Besides age and declared weight, numerical variable that is important to
consider to include in the model is the race distance. The reason to
include this variable because speed and distance has some kind of
relationship if we see it from the physicist point of view. As we
already know speed is the distance traveled per unit of time. Therefore,
if the distance becomes longer, then the speed becomes higher. In terms
of race horse, we can said that as the race distance for the race horse
to traveled become much more longer, then the speed that the race horse
require to reach that distance becomes much more faster.

Observe that in the runs.csv data we are given the position of the horse
in each section and also the time taken by the horse to complete each
section. From these data, we choose to include the time to finish the
first section into our Bayesian linear regression model due to this
variable effect the speed of the horse in each section. As the time to
finish the first section of of the race become much more longer, then
the time to complete the rest of the section become much more longer
which can cause the race horse speed to decrease since speed is not
proportional with time.

In our explanation above, we have already given the reason to choose
some continuous variable to include in our Bayesian linear regression
model. Now lets choose the categorical variable that we want to include
in the model. As we know, the race horse that win the race is clearly
have higher speed than the race horse that lose in the race. Therefore,
we should choose to include the won category since the speed of race
horse that win the race and lose in the race is different. Variable like
horse type should also be included in the model because different horses
have different speed. For example, the difference in speed between colts
and fillies or female and male horses younger than four years is only
1%. However, male castrates are more obedient and calmer than
uncastrated ones. In our previous analysis, we have discussed some
categorical variable regarding horse type by the winning category and
based on their kind. If we talk about race in general, clearly the race
track condition affecting the speed of someone who is participating on a
race in that track. As the track become much more bumpy, extreme, or
slippery then the racer must somehow reduced their speed. This also
works on horse since they must pay attention to their track that they
used. Therefore, we should consider categorical variable that related to
race track such as race venue, race surface, race configuration, and
race track condition.

In summary, the explanatory that we are going to use in our new Bayesian
linear regression model to predict the mean speed of a race horse is
race venue, track configuration, track condition, horse type, winning
category, race distance, time to finish the first section, horse age,
and declared weight. After we choose our new explanatory variable, we
will fit our new Bayesian linear regression model using our new prior
for the intercept while other hyperparameter in the regression model
remains the same.

According to our model summary below, we can see that the posterior mean
of the intercept parameter is 16.581, which implies that the average
mean speed of each horse will be equal to 16.581 if all observed values
of the explanatory variable (race venue, track configuration, track
condition, horse type, winning category, race distance, time to finish
the first section, horse age, and declared weight) in our model equals
to zero. As we can see from the model summary below, the column
$\verb|0.025quant|$ and $\verb|0.975quant|$ represents the lower and
upper bound of a 95% credible interval for our regression parameter. In
Bayesian statistics, we can interpret the 95% credible interval for the
regression parameter as a fix interval where the value of interest of
our regression parameter will lies on that interval with probability
95%.

``` r
options(max.print=100)

#add new explanatory variable to runs data
add_var <- c('race_id','venue','config','surface','going')
runs_data <- merge(runs_data,races[,add_var], by = 'race_id')

#standarized the time1 and declared weight data
runs_data$time1 <- scale(runs_data$time1)
runs_data$declared_weight <- scale(runs_data$declared_weight)

#convert categorical variable into factor
runs_data$venue <- as.factor(runs_data$venue)
runs_data$config <- as.factor(runs_data$config)
runs_data$surface <- as.factor(runs_data$surface)
runs_data$going <- as.factor(runs_data$going)
runs_data$won <- as.factor(runs_data$won)
runs_data$horse_type <- as.factor(runs_data$horse_type)

#prior for beta and precision
prec2.prior <- list(prec=list(prior="loggamma",param=c(0.1,0.1)))
prior2.beta <- list(mean.intercept=17.66,prec.intercept=0.2536206,
                    mean = 0, prec = 1e-06)

#input the regression model to INLA 
m2.I <- inla(speed~distance+horse_age+time1+declared_weight+won+
               venue+surface+going+horse_type,
             data = runs_data, family = "gaussian",
             control.predictor = list(compute = T),
             control.compute = list(cpo=T,dic = T,config = T),
             control.family = list(hyper=prec2.prior),
             control.fixed = prior2.beta)

#display model summary
summary(m2.I)
```

    ## 
    ## Call:
    ##    c("inla.core(formula = formula, family = family, contrasts = contrasts, 
    ##    ", " data = data, quantiles = quantiles, E = E, offset = offset, ", " 
    ##    scale = scale, weights = weights, Ntrials = Ntrials, strata = strata, 
    ##    ", " lp.scale = lp.scale, link.covariates = link.covariates, verbose = 
    ##    verbose, ", " lincomb = lincomb, selection = selection, control.compute 
    ##    = control.compute, ", " control.predictor = control.predictor, 
    ##    control.family = control.family, ", " control.inla = control.inla, 
    ##    control.fixed = control.fixed, ", " control.mode = control.mode, 
    ##    control.expert = control.expert, ", " control.hazard = control.hazard, 
    ##    control.lincomb = control.lincomb, ", " control.update = 
    ##    control.update, control.lp.scale = control.lp.scale, ", " 
    ##    control.pardiso = control.pardiso, only.hyperparam = only.hyperparam, 
    ##    ", " inla.call = inla.call, inla.arg = inla.arg, num.threads = 
    ##    num.threads, ", " blas.num.threads = blas.num.threads, keep = keep, 
    ##    working.directory = working.directory, ", " silent = silent, inla.mode 
    ##    = inla.mode, safe = FALSE, debug = debug, ", " .parent.frame = 
    ##    .parent.frame)") 
    ## Time used:
    ##     Pre = 0.641, Running = 4.54, Post = 12, Total = 17.2 
    ## Fixed effects:
    ##                         mean    sd 0.025quant 0.5quant 0.975quant mode kld
    ## (Intercept)           16.581 0.151     16.285   16.581     16.878   NA   0
    ## distance              -0.326 0.002     -0.330   -0.326     -0.322   NA   0
    ## horse_age              0.002 0.002     -0.002    0.002      0.006   NA   0
    ## time1                 -0.034 0.002     -0.038   -0.034     -0.029   NA   0
    ## declared_weight        0.022 0.002      0.018    0.022      0.026   NA   0
    ## won1                   0.174 0.007      0.160    0.174      0.188   NA   0
    ## venueST                0.129 0.004      0.120    0.129      0.137   NA   0
    ## surface1               0.036 0.008      0.020    0.036      0.052   NA   0
    ## goingGOOD             -0.078 0.013     -0.103   -0.078     -0.052   NA   0
    ## goingGOOD TO FIRM     -0.002 0.014     -0.029   -0.002      0.025   NA   0
    ## goingGOOD TO YIELDING -0.129 0.016     -0.160   -0.129     -0.099   NA   0
    ## goingSOFT             -0.629 0.023     -0.675   -0.629     -0.584   NA   0
    ## goingWET SLOW         -0.183 0.028     -0.238   -0.183     -0.128   NA   0
    ## goingYIELDING         -0.196 0.023     -0.242   -0.196     -0.151   NA   0
    ##  [ reached getOption("max.print") -- omitted 9 rows ]
    ## 
    ## Model hyperparameters:
    ##                                          mean    sd 0.025quant 0.5quant
    ## Precision for the Gaussian observations 21.93 0.255      21.41    21.91
    ##                                         0.975quant mode
    ## Precision for the Gaussian observations      22.54   NA
    ## 
    ## Deviance Information Criterion (DIC) ...............: -3144.63
    ## Deviance Information Criterion (DIC, saturated) ....: 415038.97
    ## Effective number of parameters .....................: 23.96
    ## 
    ## Marginal log-Likelihood:  1333.21 
    ## CPO, PIT is computed 
    ## Posterior summaries for the linear predictor and the fitted values are computed
    ## (Posterior marginals needs also 'control.compute=list(return.marginals.predictor=TRUE)')

Observe that the values of the posterior mean and standard deviation for
the posterior distribution of the horse age are close to 0. This implies
that the regression parameter is likely not important in the model.
However, since horse age is a confounding variable we will keep this
variable in our model.

In our Bayesian linear regression model summary, we are given the
descriptive statistics for the precision $\tau$. This parameter has
posterior mean and standard deviation equals to 22.01 and 0.266
respectively. In addition, we also obtain the 95% credible interval for
the precision $\tau$, which is (21.46, 22.54). Therefore, we can say
that the actual values of the precision $\tau$ will lies between 21.46
and 22.54 with probability 95%. Since in our Bayesian linear regression
model we are given given the marginal of the precision $\tau$, then we
need to obtain the marginal of the variance $\sigma^2$ by using the
$\verb|inla.tmargimal|$ function in $\verb|R|$.

Observe that in $\verb|inla.tmarginal|$ function we defined a function
that contains a formula to compute the variance $\sigma^2$ by using the
precision $\tau$. As we know, the formula that describe the relationship
between variance $\sigma^2$ and precision $\tau$ is
$\sigma^2 = \frac{1}{\tau}$. After we obtain the marginal of the
variance $\sigma^2$, we can plot the posterior density of $\sigma^2$.
According to the posterior density curve for the variance $\sigma^2$, we
can see that the posterior distribution for $\sigma^2$ is a left-skewed
distribution. In addition, we also obtain a summary statistics about the
posterior mean and standard deviation for $\sigma^2$, which is 0.0454456
and 0.000554374 respectively. Not only that, we also obtain the 2.5% and
97.5% quantile for $\sigma^2$, which is 0.0443656 and 0.0466175
respectively. As we already know, the 2.5% and 97.5% quantile for
$\sigma^2$ represents the lower and upper bound for the 95% credible
interval for the variance $\sigma^2$. Therefore, we can said that the
true values of the variance $\sigma^2$ will lies between 0.0443656 and
0.0466175 with a probability 95%.

``` r
#Summary statistics of sigma2
marg2.sigma2 <- inla.tmarginal(function(tau) tau^(-1),
                               m2.I$marginals.hyperpar[[1]])
cat("Summary statistics of sigma square: \n")
```

    ## Summary statistics of sigma square:

``` r
inla.zmarginal(marg2.sigma2)
```

    ## Mean            0.0455965 
    ## Stdev           0.000519138 
    ## Quantile  0.025 0.0443614 
    ## Quantile  0.25  0.0453083 
    ## Quantile  0.5   0.0456471 
    ## Quantile  0.75  0.0458912 
    ## Quantile  0.975 0.0467124

``` r
#Plot of marginal of sigma
plot(marg2.sigma2, type ="l",xlab=expression(sigma^2),
     ylab="Density", 
     main='Posterior density of'~sigma^2~'for model 2')
```

![](assignment-1_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
#compute NSLCPO
cat("NSLCPO:",-sum(log(m2.I$cpo$cpo)),"\n")
```

    ## NSLCPO: -1566.648

``` r
#compute DIC
cat("DIC:",m2.I$dic$dic,"\n")
```

    ## DIC: -3144.627

``` r
#compute standard deviation of mean residuals
sdmr <- sd(runs_data$speed-m2.I$summary.fitted.values$mean)
cat("Standard deviation of mean residuals for model 2:",sdmr,"\n")
```

    ## Standard deviation of mean residuals for model 2: 0.2130631

According to the model summary above, we can see that the standard
deviation of mean residual for our second model is slightly higher than
the residual standard error in a) and the standard deviation of mean
residual for our first model in b). However, we can see that the NSLCPO
and DIC for our new model are smaller than the NSLCPO and DIC in the
Bayesian linear regression model in b). In other words, our new Bayesian
linear regression model fits best on the data than the Bayesian linear
regression model in b).

**d)\[10 marks\] We are going to perform model checks to evaluate the
fit the two models in parts b) and c) on the data.**

**Compute the studentized residuals for the Bayesian regression model
from parts b) and c). Perform a simple Q-Q plot on the studentized
residuals. Plot the studentized residuals versus their index, and also
plot the studentized residuals against the posterior mean of the fitted
value (see Lecture 2). Discuss the results.**

First we compute the studentised residuals for the first Bayesian linear
regression model in b). We plot the posterior mean studentised residual
for each observation, versus the index numbering that observation. There
are obvious outlier in this plot, which is two points in the plot that
are below -10.

``` r
#First,we obtain samples from the posterior
nbsamp <-10000
speed.samp <- inla.posterior.sample(nbsamp, m.I)
```

    ## as(<dgCMatrix>, "dgTMatrix") is deprecated since Matrix 1.5-0; do as(., "TsparseMatrix") instead

``` r
#In this model the link function is the identity, so fitted 
#values are the same as the linear predictors 
#(E(y_i|x,theta)=mu_i=eta_i)
sigma <-1/sqrt(inla.posterior.sample.eval(function(...) {theta}, speed.samp))
fittedvalues <-inla.posterior.sample.eval(function(...){Predictor}, speed.samp)

n <- nrow(runs_data)
x <- cbind(rep(1,n),runs_data$distance, runs_data$horse_rating,
           runs_data$horse_age, runs_data$horse_id)
H <- x%*%solve((t(x)%*%x))%*%t(x)

#studentised residuals
#n is the number of observations
studentisedred <- matrix(0,nrow=n,ncol=nbsamp)

#create a matrix of size n * nbsamp, repeating y in each column
y <- runs_data$speed
ymx <- as.matrix(y)%*%matrix(1,nrow=1,ncol=nbsamp);

studentisedred <- ymx-fittedvalues;

for(l in 1:nbsamp){
  studentisedred[,l] <- studentisedred[,l]/sigma[l];
}

for(i in 1:n){
  studentisedred[i,] <- studentisedred[i,]/sqrt(1-H[i,i]);
}


#posterior mean of studentised residuals
studentisedredm <- numeric(n)
for(i in 1:n){
  studentisedredm[i] <- mean(studentisedred[i,])  
}



#Plot posterior mean studentised residual vs observation number.
par(mfrow=c(1,1))
plot(seq_along(studentisedredm),studentisedredm,xlab="Index",
     ylab="Bayesian studentised residual")
```

![](assignment-1_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

In the figure below we show a q-q plot based on the posterior mean
studentised residuals for our regression model in b). These lie
reasonably well on the diagonal line. Therefore, the error terms for
regression model in b) are normally distributed.

``` r
#QQ-plot
qqnorm(studentisedredm,lwd=2)
qqline(studentisedredm,col=2,lwd=2)
```

![](assignment-1_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

In the figure below we plot the studentised residual versus the
posterior mean of the fitted value of the model in b). In this figure we
are looking for any trends in the data which might suggest
heteroscedastic errors or missing terms in the model for the mean. There
are no obvious trends. Therefore, we can say that the variance for model
b) is constant and the linearity assumptions is not violated because
there is no indication of nonlinear pattern in this plot below.

``` r
#Compute posterior mean fitted values
fittedvaluesm <- numeric(n)
for(i in 1:n){
  fittedvaluesm[i] <- mean(fittedvalues[i,])
}

plot(fittedvaluesm,studentisedredm, cex.lab = 0.75,
     xlab="Fitted value (posterior mean)",
     ylab="Bayesian Studentised residual (posterior mean)")
```

![](assignment-1_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

Next we compute the studentised residuals for the second Bayesian linear
regression model in c). We plot the posterior mean studentised residual
for each observation, versus the index numbering that observation. There
are obvious outlier in this plot, which are two points below -10.

``` r
#First,we obtain samples from the posterior
nbsamp <-10000
speed.samp <- inla.posterior.sample(nbsamp, m2.I)

#In this model the link function is the identity, so fitted 
#values are the same as the linear predictors 
#(E(y_i|x,theta)=mu_i=eta_i)
sigma <-1/sqrt(inla.posterior.sample.eval(function(...) {theta}, speed.samp))
fittedvalues <-inla.posterior.sample.eval(function(...){Predictor}, speed.samp)

n <- nrow(runs_data)
x <- cbind(rep(1,n),runs_data$distance,runs_data$horse_age,
           runs_data$time1, runs_data$declared_weight,runs_data$won,
           runs_data$venue,runs_data$surface,runs_data$going,
           runs_data$horse_type)
H <- x%*%solve((t(x)%*%x))%*%t(x)

#studentised residuals
#n is the number of observations
studentisedred <- matrix(0,nrow=n,ncol=nbsamp)

#create a matrix of size n * nbsamp, repeating y in each column
y <- runs_data$speed
ymx <- as.matrix(y)%*%matrix(1,nrow=1,ncol=nbsamp);

studentisedred <- ymx-fittedvalues;

for(l in 1:nbsamp){
  studentisedred[,l] <- studentisedred[,l]/sigma[l];
}

for(i in 1:n){
  studentisedred[i,] <- studentisedred[i,]/sqrt(1-H[i,i]);
}


#posterior mean of studentised residuals
studentisedredm <- numeric(n)
for(i in 1:n){
  studentisedredm[i] <- mean(studentisedred[i,])  
}



#Plot posterior mean studentised residual vs observation number.
par(mfrow=c(1,1))
plot(seq_along(studentisedredm),studentisedredm,xlab="Index",
     ylab="Bayesian studentised residual")
```

![](assignment-1_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

In the figure below we show a q-q plot based on the posterior mean
studentised residuals for regression model in c). These lie reasonably
well on the diagonal line. Therefore, the error terms for regression
model in c) are normally distributed.

``` r
#QQ-plot
qqnorm(studentisedredm,lwd=2)
qqline(studentisedredm,col=2,lwd=2)
```

![](assignment-1_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

In the figure below we plot the studentised residual versus the
posterior mean of the fitted value of the model in c). In this figure we
are looking for any trends in the data which might suggest
heteroscedastic errors or missing terms in the model for the mean. There
are no obvious trends. Therefore, we can say that the variance for model
c) is constant and the linearity assumptions is not violated because
there is no indication of nonlinear pattern in this plot below.

``` r
#Compute posterior mean fitted values
fittedvaluesm <- numeric(n)
for(i in 1:n){
  fittedvaluesm[i] <- mean(fittedvalues[i,])
}

plot(fittedvaluesm,studentisedredm, cex.lab = 0.75,
     xlab="Fitted value (posterior mean)",
     ylab="Bayesian Studentised residual (posterior mean)")
```

![](assignment-1_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

**e)\[10 marks\] In this question, we are going to use the model you
have constructed in part c) to predict a new race, i.e. calculate the
posterior probabilities of each participating horse winning that race.
First, we load the dataset containing information about the future
race.**

``` r
race_to_predict <- read.csv(file = 'race_to_predict.csv')
race_to_predict
```

    ##   race_id       date venue race_no config surface distance going horse_ratings
    ## 1    1000 1998-09-18    ST       2    B+2       0     1400  GOOD         40-15
    ##    prize race_class sec_time1 sec_time2 sec_time3 sec_time4 sec_time5 sec_time6
    ## 1 485000          5        NA        NA        NA        NA        NA        NA
    ##   sec_time7 time1 time2 time3 time4 time5 time6 time7 place_combination1
    ## 1        NA    NA    NA    NA    NA    NA    NA    NA                  5
    ##   place_combination2 place_combination3 place_combination4 place_dividend1
    ## 1                  7                  8                 NA            27.5
    ##   place_dividend2 place_dividend3 place_dividend4 win_combination1
    ## 1              43              57              NA                5
    ##   win_dividend1 win_combination2 win_dividend2
    ## 1            86               NA            NA

``` r
runs_to_predict <- read.csv(file = 'runs_to_predict.csv')
runs_to_predict
```

    ##   race_id horse_no horse_id result won lengths_behind horse_age horse_country
    ## 1    1000        1     3940     NA  NA             NA         3            NZ
    ## 2    1000        2      474     NA  NA             NA         3            NZ
    ##   horse_type horse_rating horse_gear declared_weight actual_weight draw
    ## 1    Gelding           60         --            1148           133    7
    ## 2    Gelding           60         --            1039           122    4
    ##   position_sec1 position_sec2 position_sec3 position_sec4 position_sec5
    ## 1             9             6             6            14            NA
    ## 2             4             4             4             4            NA
    ##   position_sec6 behind_sec1 behind_sec2 behind_sec3 behind_sec4 behind_sec5
    ## 1            NA        2.75        2.75           3       11.00          NA
    ## 2            NA        1.00        1.75           2        2.25          NA
    ##   behind_sec6 time1 time2 time3 time4 time5 time6 finish_time win_odds
    ## 1          NA    NA    NA    NA    NA    NA    NA          NA     55.0
    ## 2          NA    NA    NA    NA    NA    NA    NA          NA      4.6
    ##   place_odds trainer_id jockey_id
    ## 1       17.0         38       138
    ## 2        1.7         47        31
    ##  [ reached 'max' / getOption("max.print") -- omitted 12 rows ]

**Based on your model from part c), compute the posterior probabilities
of each of these 14 horses winning the race. \[Hint: you will need to
sample from the posterior predictive distribution.\]**

In this part, we will use model in part c) to compute the posterior
probabilities of these 14 horses winning in the race. First we include a
new row in the runs data frame that we use to build our Bayesian linear
regression model in c), with the response variable(mean speed) set to
NA. After this, we fit the model in INLA using model c).

``` r
#merge the two table race and runs with horse id 1000
add_var <- c('race_id','distance','venue','config','surface','going')
runs_new <- merge(runs_to_predict,race_to_predict[,add_var], 
                  by = 'race_id')

#add the empty speed column to the new runs data
runs_new['speed'] <- rep(NA, nrow(runs_new))

#input the original runs and races data into R
runs <- read.csv(file = 'runs.csv')
races <- read.csv(file = 'races.csv')

#merge the two original runs and races data
runs_ori <- merge(runs,races[,add_var], by = 'race_id')

#add the speed column to the original runs data
runs_ori['speed'] <- runs_ori$distance/runs_ori$finish_time

#combine the original and new runs data
runs_new <- rbind(runs_new, runs_ori)

#convert categorical variable to factor
runs_new$horse_id <- as.factor(runs_new$horse_id)
runs_new$venue <- as.factor(runs_new$venue)
runs_new$config <- as.factor(runs_new$config)
runs_new$surface <- as.factor(runs_new$surface)
runs_new$going <- as.factor(runs_new$going)
runs_new$won <- as.factor(runs_new$won)
runs_new$horse_type <- as.factor(runs_new$horse_type)

#scale all numerical variable covariates that include in the model 
runs_new$distance <- scale(runs_new$distance)
runs_new$horse_age <- scale(runs_new$horse_age)
runs_new$time1 <- scale(runs_new$time1)
runs_new$declared_weight <- scale(runs_new$declared_weight)

#prior for beta and precision
prec2.prior <- list(prec=list(prior="loggamma",param=c(0.1,0.1)))
prior2.beta <- list(mean.intercept=17.66,prec.intercept=0.2536206,
                    mean = 0, prec = 1e-06)

#input the regression model to INLA 
m3.I <- inla(speed~distance+horse_age+time1+declared_weight+won+
               venue+surface+going+horse_type,
             data = runs_new, family = "gaussian",
             control.predictor = list(compute = T),
             control.compute = list(cpo=T,dic = T,config = T),
             control.family = list(hyper=prec2.prior),
             control.fixed = prior2.beta)

#display model summary
summary(m3.I)
```

    ## 
    ## Call:
    ##    c("inla.core(formula = formula, family = family, contrasts = contrasts, 
    ##    ", " data = data, quantiles = quantiles, E = E, offset = offset, ", " 
    ##    scale = scale, weights = weights, Ntrials = Ntrials, strata = strata, 
    ##    ", " lp.scale = lp.scale, link.covariates = link.covariates, verbose = 
    ##    verbose, ", " lincomb = lincomb, selection = selection, control.compute 
    ##    = control.compute, ", " control.predictor = control.predictor, 
    ##    control.family = control.family, ", " control.inla = control.inla, 
    ##    control.fixed = control.fixed, ", " control.mode = control.mode, 
    ##    control.expert = control.expert, ", " control.hazard = control.hazard, 
    ##    control.lincomb = control.lincomb, ", " control.update = 
    ##    control.update, control.lp.scale = control.lp.scale, ", " 
    ##    control.pardiso = control.pardiso, only.hyperparam = only.hyperparam, 
    ##    ", " inla.call = inla.call, inla.arg = inla.arg, num.threads = 
    ##    num.threads, ", " blas.num.threads = blas.num.threads, keep = keep, 
    ##    working.directory = working.directory, ", " silent = silent, inla.mode 
    ##    = inla.mode, safe = FALSE, debug = debug, ", " .parent.frame = 
    ##    .parent.frame)") 
    ## Time used:
    ##     Pre = 1.14, Running = 7.82, Post = 16.3, Total = 25.3 
    ## Fixed effects:
    ##                         mean    sd 0.025quant 0.5quant 0.975quant mode kld
    ## (Intercept)           16.581 0.151     16.285   16.581     16.877   NA   0
    ## distance              -0.326 0.002     -0.329   -0.326     -0.322   NA   0
    ## horse_age              0.002 0.002     -0.002    0.002      0.006   NA   0
    ## time1                 -0.034 0.002     -0.038   -0.034     -0.029   NA   0
    ## declared_weight        0.022 0.002      0.018    0.022      0.026   NA   0
    ## won1                   0.174 0.007      0.160    0.174      0.188   NA   0
    ## venueST                0.129 0.004      0.120    0.129      0.137   NA   0
    ## surface1               0.036 0.008      0.020    0.036      0.052   NA   0
    ## goingGOOD             -0.078 0.013     -0.103   -0.078     -0.052   NA   0
    ## goingGOOD TO FIRM     -0.002 0.014     -0.029   -0.002      0.025   NA   0
    ## goingGOOD TO YIELDING -0.129 0.016     -0.160   -0.129     -0.099   NA   0
    ## goingSOFT             -0.629 0.023     -0.675   -0.629     -0.584   NA   0
    ## goingWET SLOW         -0.183 0.028     -0.238   -0.183     -0.128   NA   0
    ## goingYIELDING         -0.196 0.023     -0.242   -0.196     -0.151   NA   0
    ##  [ reached getOption("max.print") -- omitted 9 rows ]
    ## 
    ## Model hyperparameters:
    ##                                          mean    sd 0.025quant 0.5quant
    ## Precision for the Gaussian observations 22.05 0.292      21.46    22.07
    ##                                         0.975quant mode
    ## Precision for the Gaussian observations      22.57   NA
    ## 
    ## Deviance Information Criterion (DIC) ...............: -3144.16
    ## Deviance Information Criterion (DIC, saturated) ....: 415039.44
    ## Effective number of parameters .....................: 24.34
    ## 
    ## Marginal log-Likelihood:  1334.10 
    ## CPO, PIT is computed 
    ## Posterior summaries for the linear predictor and the fitted values are computed
    ## (Posterior marginals needs also 'control.compute=list(return.marginals.predictor=TRUE)')

After we complete building our new Bayesian linear regression model
using the model in c) but using the new runs data that have been
combined with the original runs data, then we will generate the
posterior sample using INLA and plot the posterior predictive
probability of winning the race for each of these 14 horse. According to
the table below and the plot of the horse number versus posterior
predictive probability of winning the race for each horse, we can see
that horse number 12 have higher probability to become the champion of
the race since its posterior predictive probability is higher than any
horse in the file runs_to_predict.csv data. One important thing to note
here is we determined the probability of each horse to win the race is
by finding the posterior probability that the horse speed will be
greater than the average speed of the horse that win the race according
to our runs.csv data. The reason we choose the mean speed of the horse
that win the race as a reference value to help determine the probability
of a horse winning the race is because the mean values represent the
tendency of the race horse speed that win the race will most likely be
around that values even though some horse have higher speed than the
mean speed of the race horse that win the race.

``` r
#generate sample from posterior distribution
nbsamp <- 10000;
m3I.samp <- inla.posterior.sample(nbsamp,m3.I,
                                  selection = list(Predictor=1:14))

#Obtain the samples from the linear predictors, which is equivalent to
#the mean of the observations as the link function is the identity here

predictor.samp <- inla.posterior.sample.eval(function(...){Predictor},
m3I.samp)


#We obtain the samples from the parameter sigma using the samples 
#from the precision 

sigma.samp <- 1/sqrt(inla.posterior.sample.eval(function(...){theta},
m3I.samp))



#We obtain the posterior predictive samples by adding the Gaussian 
#noise from the likelihood to the mean (mu_i=eta_i)

post.pred.samp <- matrix(0,nrow=14,ncol=nbsamp)
for(i in 1:14){
  post.pred.samp[i,]<- predictor.samp[i,]+rnorm(nbsamp,mean=0,sd=sigma.samp)
}

#calculate the posterior predictive probability
prob_win <- c()
for (i in 1:14){
  winner_speed <- mean(runs_ori$speed[which(runs_ori$won==1)],na.rm=T)
  prob_win <- c(prob_win,mean(post.pred.samp[i,]>=winner_speed))
}

#plot posterior predictive probability vs horse number
plot(runs_to_predict$horse_no, prob_win, xlab = 'Horse Number', 
     ylab='Posterior Probability of Winning The Race',type='l',
     cex.main = 0.75, cex.lab = 0.75, 
     main='Posterior Probability of Each Horse Winning The Race')
```

![](assignment-1_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

``` r
#display posterior probability data
cbind(runs_to_predict[,c('race_id','horse_no','horse_id',
                         'horse_age','horse_country',
                         'horse_type','declared_weight',
                         'actual_weight')],
      prob_win)
```

    ##    race_id horse_no horse_id horse_age horse_country horse_type declared_weight
    ## 1     1000        1     3940         3            NZ    Gelding            1148
    ## 2     1000        2      474         3            NZ    Gelding            1039
    ## 3     1000        3     3647         3            NZ    Gelding            1064
    ## 4     1000        4      144         3           AUS    Gelding            1086
    ## 5     1000        5     3712         3           AUS    Gelding            1101
    ## 6     1000        6     3734         3           AUS    Gelding            1137
    ## 7     1000        7     1988         3           AUS    Gelding            1063
    ## 8     1000        8     3247         3           AUS    Gelding            1092
    ## 9     1000        9     4320         3            NZ    Gelding            1096
    ## 10    1000       10     1077         3            NZ    Gelding            1034
    ## 11    1000       11     3916         3           AUS    Gelding            1125
    ##    actual_weight prob_win
    ## 1            133   0.3355
    ## 2            122   0.2724
    ## 3            129   0.2868
    ## 4            131   0.2916
    ## 5            128   0.3057
    ## 6            130   0.3283
    ## 7            122   0.2818
    ## 8            126   0.2945
    ## 9            126   0.3000
    ## 10           123   0.2602
    ## 11           124   0.3173
    ##  [ reached 'max' / getOption("max.print") -- omitted 3 rows ]
