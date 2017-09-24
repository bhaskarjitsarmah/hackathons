# loading data
train <- read.csv("test_tweets.csv", stringsAsFactors = FALSE)
str(train)

# loading libraries
library(qdap)
library(tm)
library(dplyr)

# data cleaning
new_stops <- c("User", "user", stopwords("en"))

train$tweet <- train$tweet %>% 
  tolower() %>%
  removePunctuation() %>%
  stripWhitespace() %>%
  bracketX() %>%
  replace_number() %>%
  replace_abbreviation() %>%
  replace_contraction() %>%
  replace_symbol() %>%
  removeWords(new_stops) %>%
  iconv("latin1", "ASCII", sub="")

write.csv(train, file = "test.csv")
