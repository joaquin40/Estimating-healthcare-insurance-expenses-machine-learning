## --------------------------------------------------------------------------------
pacman::p_load(tidyverse, skimr, DataExplorer)


## --------------------------------------------------------------------------------
df <- read.csv("./data/insurance.csv")


## --------------------------------------------------------------------------------
skim(df)


## --------------------------------------------------------------------------------
df[duplicated(df),]


## --------------------------------------------------------------------------------
df1 <- distinct(df)


## --------------------------------------------------------------------------------
DataExplorer::plot_missing(df1)


## --------------------------------------------------------------------------------

X_dummy <- model.matrix(charges ~. -region, data = df1)[,-1]
X_region <- DataExplorer::dummify(df1,select = "region") |> 
  select(starts_with("region"))


df2 <- as_tibble(cbind(X_dummy, X_region, df1["charges"]))

# or us this from library(fastDummies)
#dummy_columns(df)



## --------------------------------------------------------------------------------
df2


## --------------------------------------------------------------------------------
library(caret)
set.seed(1)

index <- caret::createDataPartition(df2$charges, p = 0.8,list = FALSE)
train <- df2[index,]
test <- df2[-index,]


## --------------------------------------------------------------------------------
par(mfrow=c(1,2))
hist(train$charges)
hist(test$charges)


## --------------------------------------------------------------------------------
library(h2o)

h2o.init()


## --------------------------------------------------------------------------------
set.seed(1)

index_train <- caret::createDataPartition(train$charges, p = 0.8,list = FALSE)
train_set <- train[index_train,]
validation <- train[-index_train,]


## --------------------------------------------------------------------------------
#write.csv(train_set, "./train_test_dataset/train.csv")
#write.csv(validation, "./train_test_dataset/validation.csv")
#write.csv(test, "./train_test_dataset/test.csv")


## --------------------------------------------------------------------------------
train.hex <- as.h2o(train, destination_frame = "train.hex")
validate.hex <- as.h2o(validation, destination_frame = "validate.hex")

class(train.hex)


## --------------------------------------------------------------------------------
response <- "charges"
predictors <- colnames(train)
predictors <- predictors[!predictors %in% response]


## --------------------------------------------------------------------------------
model <- h2o.automl(
  x = predictors,
  y = response,
  training_frame = train.hex,
  validation_frame = validate.hex
  #,max_runtime_secs = 1200
)


## --------------------------------------------------------------------------------
(leader <- model@leader)
(rmse <- h2o.rmse(leader,train = FALSE,xval = TRUE))


## --------------------------------------------------------------------------------
test.hex <- as.h2o(test, destination_frame = "text.hex")


## --------------------------------------------------------------------------------
fit <- model@leader
pred <- h2o.predict(fit, test.hex)
performance  <- h2o.performance(fit, test.hex); performance 

glue::glue("RMSE:", h2o.rmse(performance) |> round(2), .sep = " ")
glue::glue("MAE:", h2o.mae(performance) |> round(2), .sep = " ")


tibble(as.data.frame( test.hex$charges), as.data.frame(pred$predict ))



## --------------------------------------------------------------------------------
# h2o.saveModel(fit, path = "./model/")


## --------------------------------------------------------------------------------
#loaded_model <- h2o.loadModel("model/StackedEnsemble_BestOfFamily_8_AutoML_6_20231105_143316")

