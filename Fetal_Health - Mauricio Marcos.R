# Statistical Learning: step 1

# Introduction

#Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress.
#The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births. Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.
#In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.

# Goal
# The goal of this part of the homework is to apply all statistical tools studied in order to classify, in the best way possible (what is the "best" way possible will be discussed later on), a new fetus depending on various features.
# For this, we will use Quadratic Discriminant Analysis, Linear Discriminant Analysis, Naive Bayes and Logistic Regression.

# Installing libraries
library(ggplot2)
library(MASS)
library(e1071) 
library(VGAM)
library(tidyverse)
library("MASS")
library("rrcov")
library("bestNormalize")
library("outliers")
library("EnvStats")
library("dplyr")
library("plotrix")
library("caret")
library("tidyverse")
library("penalizedLDA")


ldaFit <- train(fetal_health ~ ., 
                method = "PenalizedLDA", 
                data = FetusTrain,
                metric = "Accuracy",
                trControl = ctrl)
ldaPred = predict(ldaFit, FetusTest)
confusionMatrix(ldaPred,FetusTest$fetal_health)

#Defining Colours
color_1 <- "blue"
color_2 <- "red"
color_3 <- "yellow"

# Data Source
#This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:
#Normal
#Suspect
#Pathological


data <- read.csv("fetal_health.csv")
dim(data)
head(data)
summary(data)

# Variables Description
lapply(data,class)


# Target Variable: "fetal health"
# This is a categorical variable, as it clasifies the overall health of the fetus. 
# It takes value:
# 1- normal helath
# 2- suspect health
# 3- dangerous health

# Missing Values
# We have to check if there are any missing values in our data.

n_miss <- apply(is.na(data),1,sum)
table(n_miss)

# As we can see, there are no missing values.

# Let´s explore our response variable first.


target.freq <- table(data$fetal_health)
target.freq
target.freq.rel <- target.freq/nrow(data)
target.freq.rel

# We can see that the classes are pretty unbalanced. This may become problematic with bayesian classifiers, as it is best if classes are balanced.
# We also see there is a clear predominancie of healthy fetus, and a minority of suspect and even less pathological fetus. Actually, almost 78% of the fetus in our data are considered healthy, while almost 14% suspicious and just 8% are in danger.
# This is reasonable: is we consider this a random sample, then most of the fetus are usually healthier, while it is not so common to find suspected or problematic fetus.
# After talking with a neonathologist, I decided that merging the suspicious and dangerous fetus is not a bad idea. This way the classes will be more balanced, and in practical effects, both suspisious and dangerous fetus would have to be checked with caution.

data$fetal_health = factor(ifelse(data$fetal_health == 1, "Healthy" , "Warning"))

target.freq <- table(data$fetal_health)
target.freq
target.freq.rel <- target.freq/nrow(data)
target.freq.rel

lbles <- paste(names(target.freq), "\n", target.freq, sep="")
pie3D(target.freq,labels=lbles,explode=0.1,main="Pie Chart of Fetal Health",)

ggplot(data=data, aes(x = fetal_health)) + geom_bar(aes(fill = fetal_health)) + ggtitle("Bar Plot for Fetal Health")


features <- setdiff(colnames(data),'fetal_health')
features

# We have 21 features, described as follows.
# 1) Baseline Value: Beseline fetal heart rate (FHR)
# 2) Accelerations: Number of accelerations per second
# 3) Fetal Movement: Number of fetal movements per second
# 4) Uterine Contractions: Number of uterine contractions per second
# 5) Light Decelerations: Number of LDs per second
# 6) Severe decelerations: Number of SDs per second
# 7) Prolongued Decelerations: Number of PDs per second
# 8) Abnormal Short Term Variability: Percentage of time with abnormal short term variability
# 9) Mean Value of Short Term Variability: Mean value of short term variability
# 10) Percentage of time with Abnormal Long Term Variability: Percentage of time w/ abnormal long term variability
# 11) Mean Value of Long term Variability
# 12) Histogram Width: Width of histogram done using all values from a record
# 13) Histogram Min: histogram minimum value
# 14) Histogram Max: histogram maximum value
# 15) Histogram Number of Peaks: number of peaks in the exam histogram
# 16) Histogram Number of Zeroes: Number of zeros in the exam histogram
# 17) Histogram Mode 
# 18) Histogram Mean  
# 19) Histogram Median
# 20) Histogram Variance
# 21) Histogram Tendency: histogram trend

# Let´s explore their histograms

for (f in features) {
  hist(data[,f], main=f)
}

# Some corrections to be made
# There are some variables that get a lot (most of the observations) of zeros. For instance, most of the cases are of healthy fetus, so the "severe decelerations" are almost not seen in the data.
# The difficulty arises when we combine continuous variables with variables that take almost always the same value. To avoid this, we will convert these variables into dummies when considered possible.
# But we can not just do it in every case. This is because we still have to be able to distinguish between the serious and suspicious cases (that are very few in our observations). If we say when a variable takes any value larger than 0, it will be assigned a "1" then we are not catching these two classes, but marging them. So we will do this only for very extreme cases.
# I am interested in analyzing if fetal_decelrations, severe_decelerations, prologued_decelerations, histogram_number_of_zeros varie between suspicious and dangerous fetus. 

healthy_fetus <- X_trans[ X_trans$fetal_health=="Healthy",]
healthy_fetus <- healthy_fetus[,1:16]
summary(healthy_fetus)
D <- cor(healthy_fetus)
ggcorrplot(D,lab=TRUE)


dangerous_fetus <- data[data$fetal_health==3,]
dangerous_fetus <- dangerous_fetus[,1:21]

hist(dangerous_fetus$fetal_movement)
hist(suspect_fetus$fetal_movement)

hist(dangerous_fetus$severe_decelerations)
hist(suspect_fetus$severe_decelerations)
summary(suspect_fetus$severe_decelerations)
summary(dangerous_fetus$severe_decelerations)

hist(dangerous_fetus$prolongued_decelerations)
hist(suspect_fetus$prolongued_decelerations)

hist(dangerous_fetus$histogram_number_of_zeroes)
hist(suspect_fetus$histogram_number_of_zeroes)






# Let´s study the problem in one dimension, to see if there is any single feature that helps us distinguish between the tree classes.
dat <- data
dat$fetal_health <- as.factor(dat$fetal_health)

univar_graph <- function(univar_name, univar, data, output_var) {
  g_1 <- ggplot(data, aes(x=univar)) + geom_density() + xlab(univar_name)
  g_2 <- ggplot(data, aes(x=univar, fill=output_var)) + geom_density(alpha=0.4) + xlab(univar_name)
  grid.arrange(g_1, g_2, ncol=2, top=paste(univar_name,"variable", "/ [ Skew:",skewness(univar),"]"))
}

for (x in 1:(ncol(data)-1)) {
  univar_graph(names(data)[x], data[,x], data, dat[,'fetal_health'])
}

# It would be interesting to perform the same analysis, but in a two dimessional perspective. unfortunatly ploting 21 quantitative variables with each other would result in a combination of: (21*21)-21 = 420 scatterplots.
# So we would have to analyze 110 scatterplots to see this properly. As this is not very reasonable, one good way of studying what variables may be influential to classify our data, we can use principal component analysis.


#Data Preparation



# We have no missing value.

# Mean vector, Covariance and Correlation Matrix
#Mean Vector
X <- data
m <- colMeans(X_trans[,1:21])
m

# Correlation Matrix
R <- cor(X[,1:21])
R
ggcorrplot(R)


###########################
#PCA
X_quant <- data[,1:21]
Y <- data [,22]
X_pcs <- prcomp(X_transs,scale=TRUE)
colors_X <- c(color_2,color_3)[1*(Y=="Healthy")+1]
par(mfrow=c(1,1))
plot(X_pcs$x[,1:2],pch=19,col=colors_X)
get_eigenvalue(X_pcs)
library(factoextra)

X_transs <- X_trans[,1:16]
pca_output <- PCA(X_transs,ncp = 8, graph = FALSE)
pca_output$var$contrib
fviz_screeplot(pca_output,ncp=8,addlabels=T,barfill="blue",barcolor="red")
pca_output$eig
pca_output_ret <- paran(X_transs,seed=1,graph = TRUE)
pca_output_ret$Retained
pca_output$eig[,3][1:5]
fviz_contrib(pca_output, choice="var", axes=1,top=5)
fviz_contrib(pca_output, choice="var", axes=2,top=5)
fviz_contrib(pca_output, choice="var", axes=3,top=5)
fviz_contrib(pca_output, choice="var", axes=4,top=5)
fviz_contrib(pca_output, choice="var", axes=5,top=5)

# Check if the first two components give me any sign of subgroups.
colors_X <- c(color_2,color_3)[1*(Y=="Healthy")+1]
par(mfrow=c(1,1))
pairs(X_pcs$x[,1:5],col=colors_X,pch=19,main="The first three PCs")

# Number 3 is the one that gives us more information about the groups.



#################


X = subset(data, select = -c(histogram_median,histogram_mode,histogram_min,histogram_variance,histogram_mean,histogram_width))
dim(X)

# Let´s check for outliers
# First I will transform the variables, to make them more symetrical and close to a gaussian. Also, when diong this, sometimes some outliers dissapear, as variability shrinkages.


X_trans <- X

baseline_value_norm <- bestNormalize(X_trans$baseline.value,out_of_sample = TRUE)
baseline_value_norm
X_trans[,1] <- baseline_value_norm$x.t
colnames(X_trans)[1] <- "baseline_value_norm"

accelerations_norm <- bestNormalize(X_trans$accelerations,out_of_sample = TRUE)
accelerations_norm
X_trans[,2] <- accelerations_norm$x.t
colnames(X_trans)[2] <- "accelerations_norm"

fetal_movement_norm <- bestNormalize(X_trans$fetal_movement,out_of_sample = TRUE)
fetal_movement_norm
X_trans[,3] <- fetal_movement_norm$x.t
colnames(X_trans)[3] <- "fetal_movement_norm"


uterine_contractions_norm <- bestNormalize(X_trans$uterine_contractions)
uterine_contractions_norm
X_trans[,4] <- uterine_contractions_norm$x.t
colnames(X_trans)[4] <- "uterine_contractions_norm"

light_decelerations_norm <- bestNormalize(X_trans$light_decelerations)
light_decelerations_norm
X_trans[,5] <- light_decelerations_norm$x.t
colnames(X_trans)[5] <- "light_decelerations_norm"


severe_decelerations_norm <- bestNormalize(X_trans$severe_decelerations)
severe_decelerations_norm
X_trans[,6] <- severe_decelerations_norm$x.t
colnames(X_trans)[6] <- "severe_decelerations_norm"

prolongued_decelerations_norm <- bestNormalize(X_trans$prolongued_decelerations)
prolongued_decelerations_norm
X_trans[,7] <- prolongued_decelerations_norm$x.t
colnames(X_trans)[7] <- "prolongued_decelerations_norm"

abnormal_short_term_variability_norm <- bestNormalize(X_trans$abnormal_short_term_variability)
abnormal_short_term_variability_norm
X_trans[,8] <- abnormal_short_term_variability_norm$x.t
colnames(X_trans)[8] <- "abnormal_short_term_variability_norm"

 
mean_value_of_st_var_norm <- bestNormalize(X_trans$mean_value_of_short_term_variability)
mean_value_of_st_var_norm
X_trans[,9] <- mean_value_of_st_var_norm$x.t
colnames(X_trans)[9] <- "mean_value_of_st_var_norm"

perc_of_time_w_abn_long_term_var_norm <- bestNormalize(X_trans$percentage_of_time_with_abnormal_long_term_variability)
X_trans[,10] <- perc_of_time_w_abn_long_term_var_norm$x.t
colnames(X_trans)[10] <- "perc_of_time_w_abn_long_term_var_norm"

mean_value_of_lt_var_norm  <- bestNormalize(X_trans$mean_value_of_long_term_variability)
X_trans[,11] <- mean_value_of_lt_var_norm$x.t
colnames(X_trans)[11] <- "mean_value_of_lt_var_norm"

histogram_width_norm <- bestNormalize(X_trans$histogram_width)
X_trans[,12] <- histogram_width_norm$x.t
colnames(X_trans)[12] <- "histogram_width_norm"

histogram_max_norm <- bestNormalize(X_trans$histogram_max)
X_trans[,13] <- histogram_max_norm$x.t
colnames(X_trans)[13] <- "histogram_max_norm"

histogram_number_piks_norm <- bestNormalize(X_trans$histogram_number_of_peaks)
X_trans[,14] <- histogram_number_peaks_norm$x.t
colnames(X_trans)[14] <- "histogram_number_piks_norm"

histogram_number_zeros_norm <- bestNormalize(X_trans$histogram_number_of_zeroes)
X_trans[,15] <- histogram_number_zeros_norm$x.t
colnames(X_trans)[15] <- "histogram_number_zeros_norm"


histogram_tendency_norm <- bestNormalize(X_trans$histogram_tendency +1)
X_trans[,16] <- histogram_tendency_norm$x.t
colnames(X_trans)[16] <- "histogram_tendency_norm"

# Outlier Treatment

summary(X_trans)
dim(X_trans)
# By looking at the histograms and new summary of the variables after transformation, there are 4 variables that should be tested for outliers.
# To do this I will use the Rosner´s test. This is a very useful test, not only because it provides us with a formal test to check for outliers in each variable, but also because it can detect more than one at the same time.
# This test is also designed to detect problems related to masking: this is when two outliers are very close and one of them goes undetected. I will use k=4 as the number of suspected outliers in the variables.


test_3 <- rosnerTest(X_trans$fetal_movement_norm,k=10)
test_3$all.stats
test_6 <- rosnerTest(X_trans$severe_decelerations_norm,k=10)
test_6$all.stats
test_7 <- rosnerTest(X_trans$prolongued_decelerations_norm,k=10)
test_7$all.stats
test_16 <- rosnerTest(X_trans$histogram_number_zeros_norm,k=10)
test_16$all.stats

# Even when these tests are throwing results as outliers, I decided I am not going to get rid of them.
# The reason lies on the structure of the data. This is due to the fact that most of the fetus in the sample are healthy, therefore most of them have very standard (most of the times equal to zero) measures of several variables.
# If I take out 10 observations that represent severe cases of unhelthy fetus, I would be extracting very useful information from my data and I would make it even more unbalanced towards healthy babies.
# This way, I will interpret that these values correspond to the most serious cases, the ones that are zero (or very repetead) corresponda to healthy fetus, and the ones that are close to the healthy ones but do not represent a big jump from the mode of the variables are the ones considered suspicious. 

# Next step in our preprocessing of data is to standarize  the data.
# By standarizing we mean transforming the data to get a mean equal to zero, and standard deviation equal to one.

X_trans[1:16] <- as.data.frame(lapply(X_trans[1:16], normalize))
summary(X_trans)
dim(X_trans)

# We have now 17 variables. 16 of them are numerical, had been transformed and standarized (behave more or less like gaussians, range from 0 to 1).
# This is crucial for LDA and QDA, and specificaly standarization is crucial for LDA (as it assumes equal correlation.)


# First model


# Define the data matrix and a label (indentificator) variable
X_features <- X_trans[,1:16]
Label <- X_trans[,17]
#qda
qda.class.datos <- qda(fetal_health ~ .,X_trans)
qda.class.datos

pred.qda = predict(qda.class.datos, X_trans)$class
head(pred.qda)

post.prob.qda <- predict(qda.class.datos, X_trans)$posterior
head(post.prob.qda)


#lda
lda.class.datos <- lda(fetal_health ~ .,X_trans)
lda.class.datos

pred.lda = predict(lda.class.datos, X_trans)$class
head(pred.lda)

post.prob.lda <- predict(lda.class.datos, X_trans)$posterior
head(post.prob.lda)


# Scatterplot for predictions
colors.qda.health <- c("blue","green","orange")[pred.qda]
pairs(X_trans,main="Classification of Fetus Health",pch=19,col=colors.qda.health,lower.panel=NULL)
# Seems QDA is doing a good job, but difficult to quantify

# Same plot but with good and bad predictions (black has bad predictions and read are good ones)
colors.qda.health.good.bad <- c("black","red")[1*(Label==pred.qda)+1]
pairs(X_trans,main="Bad (in black) classifications for Fetus Health with QDA",pch=19,col=colors.qda.health.good.bad,lower.panel=NULL)

colors.lda.health.good.bad <- c("black","red")[1*(Label==pred.lda)+1]
pairs(X_trans,main="Bad (in black) classifications for Iris flowers with LDA",pch=19,col=colors.qda.iris.good.bad,lower.panel=NULL)


# summarize accuracy (confusion matrix)
# predictions in rows, true values in columns (but we can change the order)
ConfMat.qda = table(pred.qda, X_trans$fetal_health)
ConfMat.qda

error.qda <- (n - sum(diag(ConfMat.qda))) / n
error.qda

ConfMat.lda = table(pred.lda, X_trans$fetal_health)
ConfMat.lda

error.lda <- (n - sum(diag(ConfMat.lda))) / n
error.lda 

# Cross-validation is indeed coded inside LDA (no need of previous loop):
lda.class.datos <- lda(fetal_health ~ ., X_trans, CV=TRUE)

# Confusion matrix
ConfMat = table(Label, lda.class.datos$class)
ConfMat

# Computation of the proportion of errors
prop.errors <- (n - sum(diag(ConfMat))) / n
prop.errors


# Check this with train and test sets, looping to repeat the experiment 100 times.
# This is a good idea, because by only calculating this we depend a lot on luck, as we depend on many things, including what is the training and testing set that has been determined.
# In order to get a more robust idea of what is our error in predicting, we can repeat the experiment 100 times and check its distribution.

err.out=matrix(NA,nrow=20,ncol=3)
for(i in 1:20){
  train <- sample(1:2126, 2126*0.8)  # 80% of the sample for training
  n.test = 2126*0.2
  table(X_trans$fetal_health[train])   # this is the training set
  
  # train
  lda.class <- lda(fetal_health ~ ., X_trans, subset = train)
  qda.class <- qda(fetal_health ~ ., X_trans, subset = train)
  naive.class <- naiveBayes(fetal_health ~ ., X_trans, subset = train)
  
  # predict
  pred.lda = predict(lda.class, X_trans[-train, ])$class
  pred.qda = predict(qda.class, X_trans[-train, ])$class
  pred.naive = predict(naive.class, X_trans[-train, ])
  
  # test
  ConfMat.lda = table(pred.lda, X_trans$fetal_health[-train])
  err.lda <- (n.test - sum(diag(ConfMat.lda))) / n.test
  ConfMat.qda = table(pred.qda, X_trans$fetal_health[-train])
  err.qda <- (n.test - sum(diag(ConfMat.qda))) / n.test
  ConfMat.naive = table(pred.naive, X_trans$fetal_health[-train])
  err.naive <- (n.test - sum(diag(ConfMat.naive))) / n.test
  
  err.out[i,1] <- err.lda
  err.out[i,2] <- err.qda
  err.out[i,3] <- err.naive
  
}

boxplot(err.out,col="blue",xlab="", ylab="testing error", names=c("lda","qda","naive"))

# average error
apply(err.out,2,mean)





###
#última clase, vamos con caret.

spl = createDataPartition(X_trans$fetal_health, p = 0.8, list = FALSE)  # 80% for training

FetusTrain = X_trans[spl,]
FetusTest = X_trans[-spl,]

table(FetusTrain$fetal_health)






R <- cor(X_trans[,1:19])
R
ggcorrplot(R,lab=TRUE)


lr_imp <- varImp(lrFit, scale = F)
plot(lr_imp, scales = list(y = list(cex = .40)))



