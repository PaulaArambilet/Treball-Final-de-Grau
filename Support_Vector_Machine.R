library(readr)
library(tidyverse)
library(caret)
library(boot)
library(lattice)
library(ggplot2)
library(lmtest)
library(RANN)
library(class)
library(e1071)
library(shapper)
library(glmnet)
library(nnet)
library(neuralnet)
library(iml)

UK<-read.csv("C:/Users/User/Documents/Uni/TFG/Data/UK_DATA_raw_22Aug2018.csv")
UK_yoy1<-read.csv("C:/Users/User/Documents/Uni/TFG/Data/UK_macro_yoy_normed.csv")
US<-read.csv("C:/Users/User/Documents/Uni/TFG/Data/US_DATA_raw_30May2018.csv")
US_yoy1<-read.csv("C:/Users/User/Documents/Uni/TFG/Data/US_macro_yoy_normed.csv")
UK_yoy<-UK_yoy1[,-1]
US_yoy<-US_yoy1[,-1]

#Entrenem els models:

set.seed(123)
n<- nrow(UK_yoy)
n_train<-round(n*0.7)
train2<- sample(1:n, size=n_train)
train_UK<- UK_yoy[train2,]
test_UK<-UK_yoy[-train2,]

n2<- nrow(US_yoy)
n_train2<-round(n2*0.7)
train22<- sample(1:n2, size=n_train2)
train_US<- US_yoy[train22,]
test_US<-US_yoy[-train22,]


SVM_Unemployment_UK<-svm(Unemployment~.,data=train_UK,kernel="radial", gamma=0.0001,
                         cost=10, epsilon=0.0)
SVM_Unemployment_UK

pred_svm<-predict(SVM_Unemployment_UK,newdata=test_UK)
rmse(pred_svm,test_UK$Unemployment)

SVM_Inflation_UK<-svm(Inflation~.,data=train_UK,kernel="radial", gamma=0.0001,
                      cost=10, epsilon=0.0)

pred_svm_inflation<-predict(SVM_Inflation_UK,newdata=test_UK)
rmse(pred_svm_inflation,test_UK$Inflation)

SVM_GDP_UK<-svm(GDP~.,data=train_UK,kernel="radial", gamma=0.0001,
                cost=10, epsilon=0.0)

pred_svm_GDP<-predict(SVM_GDP_UK,newdata=test_UK)
rmse(pred_svm_GDP,test_UK$GDP)

#Per US
SVM_Unemployment_US<-svm(Unemployment~.,data=train_US,kernel="radial", gamma=0.0001,
                         cost=10, epsilon=0.0)

pred_svm<-predict(SVM_Unemployment_US,newdata=test_US)
rmse(pred_svm,test_US$Unemployment)

SVM_Inflation_US<-svm(Inflation~.,data=train_US,kernel="radial", gamma=0.0001,
                      cost=10, epsilon=0.0)

pred_svm_inflation<-predict(SVM_Inflation_US,newdata=test_US)
rmse(pred_svm_inflation,test_US$Inflation)

SVM_GDP_US<-svm(GDP~.,data=train_US,kernel="radial", gamma=0.0001,
                cost=10, epsilon=0.0)

pred_svm_GDP<-predict(SVM_GDP_US,newdata=test_US)
rmse(pred_svm_GDP,test_US$GDP)

##### SHAPLEY VALUES ######
process_variable <- function(data,variable_name) {
  variable_data <- filter(data, feature == variable_name)
  variable_value <- variable_data$feature.value
  variable_value_numeric <- as.numeric(gsub(paste0(variable_name, "="), "", variable_value))
  
  variable_data_processed <- mutate(variable_data, feature_value = variable_value_numeric, mean_value = mean(phi), stdfvalue = std1(phi))
  variable_data_processed <- select(variable_data_processed, -feature.value)
  variable_data_processed
}
std1 <- function(x){
  return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}
plot.shap.summary <- function(data_long){
  x_bound <- max(abs(data_long$phi))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = reorder(feature,-phi), y = phi, color = feature_value)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("feature", "mean_value")]),
              aes(x = feature, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(-3,3), labels=c("Low","High")) +
    theme_bw() + 
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom") + 
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$feature)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
  return(plot1)
}
Shapley<-function(model,target_variable,pais){
  if(target_variable=="Unemployment"){
    if(pais=="UK"){
      UK_NO_Unemployment<-UK_yoy[which(names(UK_yoy) != "Unemployment")]
      predictor_UK_Un<-Predictor$new(model, data = UK_NO_Unemployment, y = UK_yoy$Unemployment)
      Shap_UK_Un<- vector("list", nrow(UK_NO_Unemployment))
      system.time({
        for (i in seq_along(Shap_UK_Un)) {
          set.seed(123)
          Shap_UK_Un[[i]] <- iml::Shapley$new(predictor_UK_Un, x.interest = UK_NO_Unemployment[i, ],
                                              sample.size = 30)$results
          Shap_UK_Un[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_UK_Un <- dplyr::bind_rows(Shap_UK_Un)  # collapse the list.
      })
      GDP_data <- process_variable(Shap_values_UK_Un,"GDP")
      Labour_data <- process_variable(Shap_values_UK_Un,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_UK_Un,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_UK_Un,"Private_debt")
      GDHI_data <- process_variable(Shap_values_UK_Un,"GDHI")
      Inflation_data <- process_variable(Shap_values_UK_Un,"Inflation")
      Policy_rate_data <- process_variable(Shap_values_UK_Un,"Policy_rate")
      CA_data <- process_variable(Shap_values_UK_Un,"CA")
      ERI_data <- process_variable(Shap_values_UK_Un,"ERI")
      
      Shapley_UK_UN<- rbind(GDP_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Inflation_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_UK_UN<-plot.shap.summary(Shapley_UK_UN)
      return(plot_SVM_UK_UN)
    }
    if(pais=="US"){
      US_NO_Unemployment<-US_yoy[which(names(US_yoy) != "Unemployment")]
      predictor_US_Un<-Predictor$new(model, data = US_NO_Unemployment, y = US_yoy$Unemployment)
      Shap_US_Un<- vector("list", nrow(US_NO_Unemployment))
      system.time({
        for (i in seq_along(Shap_US_Un)) {
          set.seed(123)
          Shap_US_Un[[i]] <- iml::Shapley$new(predictor_US_Un, x.interest = US_NO_Unemployment[i, ],
                                              sample.size = 30)$results
          Shap_US_Un[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_US_Un <- dplyr::bind_rows(Shap_US_Un)  # collapse the list.
      })
      GDP_data <- process_variable(Shap_values_US_Un,"GDP")
      Labour_data <- process_variable(Shap_values_US_Un,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_US_Un,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_US_Un,"Private_debt")
      GDHI_data <- process_variable(Shap_values_US_Un,"GDHI")
      Inflation_data <- process_variable(Shap_values_US_Un,"Inflation")
      Policy_rate_data <- process_variable(Shap_values_US_Un,"Policy_rate")
      CA_data <- process_variable(Shap_values_US_Un,"CA")
      ERI_data <- process_variable(Shap_values_US_Un,"ERI")
      
      Shapley_US_UN<- rbind(GDP_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Inflation_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_US_UN<-plot.shap.summary(Shapley_US_UN)
      return(plot_SVM_US_UN)
    }
  }
  if(target_variable=="Inflation"){
    if(pais=="UK"){
      UK_NO_Inflation<-UK_yoy[which(names(UK_yoy) != "Inflation")]
      predictor_UK_Inf<-Predictor$new(model, data = UK_NO_Inflation, y = UK_yoy$Inflation)
      Shap_UK_Inf<- vector("list", nrow(UK_NO_Inflation))
      system.time({
        for (i in seq_along(Shap_UK_Inf)) {
          set.seed(123)
          Shap_UK_Inf[[i]] <- iml::Shapley$new(predictor_UK_Inf, x.interest = UK_NO_Inflation[i, ],
                                               sample.size = 30)$results
          Shap_UK_Inf[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_UK_Inf<- dplyr::bind_rows(Shap_UK_Inf)  # collapse the list.
      })
      GDP_data <- process_variable(Shap_values_UK_Inf,"GDP")
      Labour_data <- process_variable(Shap_values_UK_Inf,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_UK_Inf,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_UK_Inf,"Private_debt")
      GDHI_data <- process_variable(Shap_values_UK_Inf,"GDHI")
      Unemployment_data <- process_variable(Shap_values_UK_Inf,"Unemployment")
      Policy_rate_data <- process_variable(Shap_values_UK_Inf,"Policy_rate")
      CA_data <- process_variable(Shap_values_UK_Inf,"CA")
      ERI_data <- process_variable(Shap_values_UK_Inf,"ERI")
      
      Shapley_UK_Inf<- rbind(GDP_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Unemployment_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_UK_Inf<-plot.shap.summary(Shapley_UK_Inf)
      return(plot_SVM_UK_Inf)
    }
    if(pais=="US"){
      US_NO_Inflation<-US_yoy[which(names(US_yoy) != "Inflation")]
      predictor_US_Inf<-Predictor$new(model, data = US_NO_Inflation, y = US_yoy$Inflation)
      Shap_US_Inf<- vector("list", nrow(US_NO_Inflation))
      system.time({
        for (i in seq_along(Shap_US_Inf)) {
          set.seed(123)
          Shap_US_Inf[[i]] <- iml::Shapley$new(predictor_US_Inf, x.interest = US_NO_Inflation[i, ],
                                               sample.size = 30)$results
          Shap_US_Inf[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_US_Inf<- dplyr::bind_rows(Shap_US_Inf)  # collapse the list.
      })
      GDP_data <- process_variable(Shap_values_US_Inf,"GDP")
      Labour_data <- process_variable(Shap_values_US_Inf,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_US_Inf,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_US_Inf,"Private_debt")
      GDHI_data <- process_variable(Shap_values_US_Inf,"GDHI")
      Unemployment_data <- process_variable(Shap_values_US_Inf,"Unemployment")
      Policy_rate_data <- process_variable(Shap_values_US_Inf,"Policy_rate")
      CA_data <- process_variable(Shap_values_US_Inf,"CA")
      ERI_data <- process_variable(Shap_values_US_Inf,"ERI")
      
      Shapley_US_Inf<- rbind(GDP_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Unemployment_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_US_Inf<-plot.shap.summary(Shapley_US_Inf)
      return(plot_SVM_US_Inf)
    }
  }
  if(target_variable=="GDP"){
    if(pais=="UK"){
      UK_NO_GDP<-UK_yoy[which(names(UK_yoy) != "GDP")]
      predictor_UK_GDP<-Predictor$new(model, data = UK_NO_GDP, y = UK_yoy$GDP)
      Shap_UK_GDP<- vector("list", nrow(UK_NO_GDP))
      system.time({
        for (i in seq_along(Shap_UK_GDP)) {
          set.seed(123)
          Shap_UK_GDP[[i]] <- iml::Shapley$new(predictor_UK_GDP, x.interest = UK_NO_GDP[i, ],
                                               sample.size = 30)$results
          Shap_UK_GDP[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_UK_GDP<- dplyr::bind_rows(Shap_UK_GDP)  # collapse the list.
      })
      Inflation_data <- process_variable(Shap_values_UK_GDP,"Inflation")
      Labour_data <- process_variable(Shap_values_UK_GDP,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_UK_GDP,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_UK_GDP,"Private_debt")
      GDHI_data <- process_variable(Shap_values_UK_GDP,"GDHI")
      Unemployment_data <- process_variable(Shap_values_UK_GDP,"Unemployment")
      Policy_rate_data <- process_variable(Shap_values_UK_GDP,"Policy_rate")
      CA_data <- process_variable(Shap_values_UK_GDP,"CA")
      ERI_data <- process_variable(Shap_values_UK_GDP,"ERI")
      
      Shapley_UK_GDP<- rbind(Inflation_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Unemployment_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_UK_GDP<-plot.shap.summary(Shapley_UK_GDP)
      return(plot_SVM_UK_GDP)
    }
    if(pais=="US"){
      US_NO_GDP<-US_yoy[which(names(US_yoy) != "GDP")]
      predictor_US_GDP<-Predictor$new(model, data = US_NO_GDP, y = US_yoy$GDP)
      Shap_US_GDP<- vector("list", nrow(US_NO_GDP))
      system.time({
        for (i in seq_along(Shap_US_GDP)) {
          set.seed(123)
          Shap_US_GDP[[i]] <- iml::Shapley$new(predictor_US_GDP, x.interest = US_NO_GDP[i, ],
                                               sample.size = 30)$results
          Shap_US_GDP[[i]]$sample_num <- i  # identifier to track our instances.
        }
        Shap_values_US_GDP<- dplyr::bind_rows(Shap_US_GDP)  # collapse the list.
      })
      Inflation_data <- process_variable(Shap_values_US_GDP,"Inflation")
      Labour_data <- process_variable(Shap_values_US_GDP,"Labour_prod")
      Broad_money_data <- process_variable(Shap_values_US_GDP,"Broad_money")
      Private_debt_data <- process_variable(Shap_values_US_GDP,"Private_debt")
      GDHI_data <- process_variable(Shap_values_US_GDP,"GDHI")
      Unemployment_data <- process_variable(Shap_values_US_GDP,"Unemployment")
      Policy_rate_data <- process_variable(Shap_values_US_GDP,"Policy_rate")
      CA_data <- process_variable(Shap_values_US_GDP,"CA")
      ERI_data <- process_variable(Shap_values_US_GDP,"ERI")
      
      Shapley_US_GDP<- rbind(Inflation_data, Labour_data, Broad_money_data, Private_debt_data, GDHI_data, Unemployment_data, Policy_rate_data, CA_data, ERI_data)
      plot_SVM_US_GDP<-plot.shap.summary(Shapley_US_GDP)
      return(plot_SVM_US_GDP)
    }
  }
}

Shapley(SVM_Unemployment_UK,"Unemployment","UK")
Shapley(SVM_Inflation_UK,"Inflation","UK")
Shapley(SVM_GDP_UK,"GDP","UK")
Shapley(SVM_Unemployment_US,"Unemployment","US")
Shapley(SVM_Inflation_US,"Inflation","US")
Shapley(SVM_GDP_US,"GDP","US")
