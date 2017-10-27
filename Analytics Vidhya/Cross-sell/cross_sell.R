# loading libraries
library(dplyr)
library(dummies)

# loading the dataset
bog <- read.csv("/home/bhaskarjit/train.csv")

# data summary
str(bog)

# removing unnecessary variables
bog <- bog %>% select(-c(OCCUP_ALL_NEW, EEG_TAG, EEG_CLOSED, PM_FD_MON_02, PM_FD_MON_04,
                      STMT_CON_DAE_ACTIVE_MON_01, STMT_CON_DAE_CLOSED_MON_01, MER_EMI_CLOSED_MON_01,
                      MATURITY_GL, MATURITY_LAP, MATURITY_LAS))

# missing value imputation
bog$GENDER[which(bog$GENDER == "")] <- "M"
bog$GENDER <- droplevels(bog$GENDER)

# reordering levels to replace "" by "N"
level_impute <- function(data, range=8:44) {
  for(names in names(data)[range]) {
    levels(data[[names]]) <- c("N", "Y")
  }
  return(data)
}

# fill continuous NA's
fill_cont_NA <- function(data, range=45:57) {
  for(names in names(data)[range]) {
    data[[names]][which(is.na(data[[names]]) == TRUE)] <- 0
  }
  return(data)
}

# fill categorical NA's
fill_cat_NA <- function(data, range=59:76) {
  for(names in names(data)[range]) {
    data[[names]][which(is.na(data[[names]]) == TRUE)] <- 0
    data[[names]] <- factor(data[[names]])
  }
  return(data)
}

# fill continuous monthly balances
fill_amb <- function(data) {
  for(names in names(data)[87:90]) {
    data[[names]][which(is.na(data[[names]]) == TRUE)] <- median(data[[names]], na.rm = TRUE)
  }
  return(data)
}

bog <- bog %>% 
  level_impute() %>%
  fill_cont_NA() %>%
  fill_cat_NA() %>%
  level_impute(range = c(77:85, 260:261, 308:309, 313:319)) %>%
  fill_amb() %>%
  fill_cat_NA(range = c(91:93, 262:282)) %>%
  fill_cont_NA(range = c(94:202, 213:242, 244:245, 247:248, 250:251, 253:254, 256:257,
                         285:294, 305:307, 311:312, 320))

bog$COC_ELIGIBLE <- factor(bog$COC_ELIGIBLE)
bog$CRED_NEED_SCORE[which(is.na(bog$CRED_NEED_SCORE))] <- median(bog$CRED_NEED_SCORE, 
                                                                 na.rm = TRUE)

# outlier treatment
bog$AGE[which(bog$AGE == 117 | bog$AGE == 217)] <- median(bog$AGE)

# dummy variable creation
#bog <- dummy.data.frame(bog, names = c("LEGAL_ENTITY", "PA_PQ_TAG", "DESIGNATION_FINAL",
#                                       "NEFT_CC_CATEGORY", "NEFT_DC_CATEGORY",
#                                       "TPT_DC_CATEGORY_MON_01", "TPT_CC_CATEGORY_MON_01",
#                                       "IMPS_CC_CATEGORY_MON_01"), sep="_")

# feature extraction using stepwise logistic regression model
model <- glm(RESPONDERS ~ . -(CUSTOMER_ID + ZIP_CODE_FINAL + LEGAL_ENTITY + PA_PQ_TAG + 
                                DESIGNATION_FINAL + NEFT_CC_CATEGORY + NEFT_DC_CATEGORY +
                                TPT_DC_CATEGORY_MON_01 + TPT_CC_CATEGORY_MON_01 + 
                                IMPS_CC_CATEGORY_MON_01), data = bog, family = binomial)

# modelling
bog <- bog %>% select(ACT_TYPE, GENDER, PL_TAG, PL_CLOSED, CC_HOLD_MON_01, DC_HOLD, 
                      CR_LIMIT, COC_ELIGIBLE, PL_SCRUB_LIVE, PL_SCRUB_CLOSED, TWL_SCRUB_CLOSED,
                      COC_ACTIVE_MON_01, MER_EMI_ACTIVE_MON_01, TRN_CON_DAE_ACTIVE_MON_01,
                      NB_MON_01_CNT, DC_TXN_MON_06, RESPONDERS)

# modelling
library(h2o)
h2o.init(nthreads = -1)

# converting the data into h2o format
bog_hf <- as.h2o(bog)

# splitting the data
splits <- h2o.splitFrame(bog_hf, 
                         ratios = c(0.4, 0.4), 
                         seed = 42)

train_unsupervised  <- splits[[1]]
train_supervised  <- splits[[2]]
test <- splits[[3]]

response <- "RESPONDERS"
features <- setdiff(colnames(train_unsupervised), response)

# Autoencoding
bog_model_nn <- h2o.deeplearning(x = features,
                             training_frame = train_unsupervised,
                             model_id = "bog_model_nn",
                             autoencoder = TRUE,
                             ignore_const_cols = FALSE,
                             seed = 42,
                             hidden = c(10, 10, 10), 
                             epochs = 100,
                             activation = "Tanh")

h2o.saveModel(bog_model_nn, path="bog_model_nn", force = TRUE)
bog_model_nn <- h2o.loadModel("/home/bhaskarjit/bog_model_nn/bog_model_nn")
bog_model_nn

# pre-trained supervised model
train_supervised[, "RESPONDERS"] <- as.factor(train_supervised[, "RESPONDERS"])

bog_model_nn_2 <- h2o.deeplearning(y = response,
                               x = features,
                               training_frame = train_supervised,
                               pretrained_autoencoder  = "bog_model_nn",
                               balance_classes = TRUE,
                               ignore_const_cols = FALSE,
                               seed = 42,
                               hidden = c(10, 10, 10), 
                               epochs = 100,
                               activation = "Tanh")

h2o.saveModel(bog_model_nn_2, path="bog_model_nn_2", force = TRUE)
bog_model_nn_2 <- h2o.loadModel("bog_model_nn_2/DeepLearning_model_R_1509001384621_1")
bog_model_nn_2

# predictions
pred <- as.data.frame(h2o.predict(object = bog_model_nn_2, newdata = test)) %>%
  mutate(actual = as.vector(test[, 17]))

pred %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n)) 

pred %>%
  ggplot(aes(x = actual, fill = predict)) +
  geom_bar() +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap( ~ actual, scales = "free", ncol = 2)

# scoring test data
bog_test <- read.csv("/home/bhaskarjit/test.csv")

# removing unnecessary variables
bog_test <- bog_test %>% select(-c(OCCUP_ALL_NEW, EEG_TAG, EEG_CLOSED, PM_FD_MON_02, PM_FD_MON_04,
                         STMT_CON_DAE_ACTIVE_MON_01, STMT_CON_DAE_CLOSED_MON_01, MER_EMI_CLOSED_MON_01,
                         MATURITY_GL, MATURITY_LAP, MATURITY_LAS))

# missing value imputation
table(bog_test$GENDER)
bog_test$GENDER[which(bog_test$GENDER == "")] <- "M"
bog_test$GENDER <- droplevels(bog_test$GENDER)

bog_test <- bog_test %>% 
  level_impute() %>%
  fill_cont_NA() %>%
  fill_cat_NA() %>%
  level_impute(range = c(77:85, 260:261, 308:309, 313:319)) %>%
  fill_amb() %>%
  fill_cat_NA(range = c(91:93, 262:282)) %>%
  fill_cont_NA(range = c(94:202, 213:242, 244:245, 247:248, 250:251, 253:254, 256:257,
                         285:294, 305:307, 311:312, 320))

bog_test$COC_ELIGIBLE <- factor(bog_test$COC_ELIGIBLE)
bog_test$CRED_NEED_SCORE[which(is.na(bog_test$CRED_NEED_SCORE))] <- median(bog_test$CRED_NEED_SCORE, 
                                                                 na.rm = TRUE)

# outlier treatment
bog_test$AGE[which(bog_test$AGE == 117 | bog_test$AGE == 217)] <- median(bog_test$AGE)

customer_id <- bog_test$CUSTOMER_ID

bog_test <- bog_test %>% select(ACT_TYPE, GENDER, PL_TAG, PL_CLOSED, CC_HOLD_MON_01, DC_HOLD, 
                      CR_LIMIT, COC_ELIGIBLE, PL_SCRUB_LIVE, PL_SCRUB_CLOSED, TWL_SCRUB_CLOSED,
                      COC_ACTIVE_MON_01, MER_EMI_ACTIVE_MON_01, TRN_CON_DAE_ACTIVE_MON_01,
                      NB_MON_01_CNT, DC_TXN_MON_06)

# converting into h2oframe
bog_test_hf <- as.h2o(bog_test)

pred_bog <- as.data.frame(h2o.predict(object = bog_model_nn_2, newdata = bog_test_hf)) %>%
  mutate(CUSTOMER_ID=customer_id)

write.csv(pred_bog, file = "submission.csv")
