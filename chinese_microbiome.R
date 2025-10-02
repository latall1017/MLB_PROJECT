library('PLNmodels')
library(corrplot)
library(dplyr)
library(tidyverse)

# Individus en colonnes, OTU en lignes
matrix_otu <- read.table("chinese_microbiome/clean_data.txt")

meta <- read.csv("chinese_microbiome/clean_meta.txt",na.strings=c(" "),sep="\t",row.names=1)
taxo <- read.csv("chinese_microbiome/clean_tax.txt",na.strings=c(" "),sep="\t")


# on prépare les données pour PLN
matrix_otu <- as.data.frame(t(matrix_otu))
row.names(meta) <- row.names(matrix_otu)
data_PLN <- list(matrix_otu=matrix_otu,morph_data=meta)
data_PLN <- prepare_data(counts = data_PLN$matrix_otu, covariates = data_PLN$morph_data)


######### Modélisation PLN ###############

#sans covariable ni offset-----------------
myPLN <- PLN(Abundance ~ 1, data_PLN)

fitted   = as.vector(fitted(myPLN))
observed = as.vector(data_PLN$Abundance)

plot(observed,fitted) 
summary(lm(fitted~observed))    #trop de paramètres
myPLN$criteria

# Sigma
heatmap(myPLN$model_par$Sigma,Colv = NA, Rowv = NA)

# sans covariable et avec offset : TSS--------------
myPLN_offsets <- PLN(Abundance ~ 1 + offset(log(Offset)),data = data_PLN)
fitted   = as.vector(fitted(myPLN_offsets))
observed = as.vector(data_PLN$Abundance)

plot(observed,fitted) 
summary(lm(fitted~observed))    


rbind(
  myPLN$criteria,
  myPLN_offsets$criteria
) %>% knitr::kable()


# Avec covariable weight-------------------
myPLN_weight <- PLN(Abundance ~ 1 + Weight, data = data_PLN)

fitted   = as.vector(fitted(myPLN_weight))
observed = as.vector(data_PLN$Abundance)

plot(observed,fitted) 
summary(lm(fitted~observed))    

rbind(
  myPLN$criteria,
  myPLN_weight$criteria
) %>% knitr::kable()


# Theta^
heatmap(myPLN_weight$model_par$Theta,Colv = NA, Rowv = NA)
# Sigma^
heatmap(myPLN_weight$model_par$Sigma,Colv = NA, Rowv = NA)
#MU^
un <- rep(1,14)
X <- matrix(c(un,data_PLN$Weight),nrow=14)
mu_chap <- X%*%t(myPLN_weight$model_par$Theta)+(un%*%t(diag(myPLN_weight$model_par$Sigma)))/2
# on regarde les corrélations entre OTU dans cette matrice mu_chap
heatmap(cor(mu_chap),Colv = NA, Rowv = NA)
heatmap(cor(matrix_otu_D0),Colv = NA, Rowv = NA)




### On force la matrice sigma comme étant diagonale
myPLN_diag <- PLN(Abundance ~ 1, data_PLN,control = list(covariance = "diagonal"))

fitted   = as.vector(fitted(myPLN_diag))
observed = as.vector(data_PLN$Abundance)

plot(observed,fitted) 
summary(lm(fitted~observed))    

rbind(
  myPLN$criteria,
  myPLN_diag$criteria
) %>% knitr::kable()



###### PCA ################
## Modèle sans covariable
myPCA_m0 <- PLNPCA(formula = Abundance ~ 1 + offset(log(Offset)),
                   data = data_PLN, 
                   ranks = 1:3) 
myPCA_m0
plot(myPCA_m0, reverse = TRUE)
#on extrait meilleur modèle (selon BIC)
PCA_m0_BIC <- getBestModel(myPCA_m0, "BIC")
PCA_m0_BIC                       

#le graphe des individus : très intéssant, mais il faudrait comprendre commenton arrive à ca!!!!
factoextra::fviz_pca_ind(PCA_m0_BIC,col.ind = data_PLN$Sex,repel=T)
factoextra::fviz_pca_ind(PCA_m0_BIC,col.ind = data_PLN$Age,repel=T,gradient.cols = c("yellow","red"))
factoextra::fviz_pca_ind(PCA_m0_BIC,col.ind = data_PLN$Group,repel=T)
factoextra::fviz_pca_ind(PCA_m0_BIC,col.ind = as.numeric(data_PLN$Weight_kg),repel=T,gradient.cols = c("yellow","red"))








