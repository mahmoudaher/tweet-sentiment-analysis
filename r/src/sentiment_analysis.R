# Train and evaluate a tweet sentiment classifier with R.

required_packages <- c(
  "caret", "dplyr", "ggplot2", "readr", "stringr", "tidyr",
  "tidytext", "wordcloud", "e1071", "RColorBrewer"
)
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Install the missing packages before running: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(stringr)
  library(tidyr)
  library(tidytext)
  library(wordcloud)
  library(e1071)
  library(RColorBrewer)
})

args <- commandArgs(trailingOnly = TRUE)
dataset_path <- if (length(args) >= 1) args[[1]] else "data/Tweets.csv"
output_dir <- if (length(args) >= 2) args[[2]] else "outputs/r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(dataset_path)) {
  stop("Dataset not found: ", dataset_path, call. = FALSE)
}

data <- read_csv(dataset_path, locale = locale(encoding = "Latin1"), show_col_types = FALSE) %>%
  select(text, sentiment) %>%
  rename(tweet = text, label = sentiment) %>%
  filter(!is.na(tweet), !is.na(label), nchar(tweet) > 5) %>%
  mutate(
    label = factor(str_to_lower(str_trim(label))),
    document_id = row_number(),
    clean_text = tweet %>%
      str_to_lower() %>%
      str_replace_all("https?://\\S+|www\\.\\S+", " ") %>%
      str_replace_all("@\\w+", " ") %>%
      str_replace_all("#", " ") %>%
      str_replace_all("[^a-zA-Z\\s]", " ") %>%
      str_squish()
  ) %>%
  filter(nchar(clean_text) > 0)

tokens <- data %>%
  select(document_id, label, clean_text) %>%
  unnest_tokens(word, clean_text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_length(word) > 1)

word_counts <- tokens %>% count(document_id, word, name = "term_frequency")
term_matrix <- word_counts %>%
  pivot_wider(names_from = word, values_from = term_frequency, values_fill = 0) %>%
  left_join(data %>% select(document_id, label), by = "document_id")

set.seed(42)
train_rows <- createDataPartition(term_matrix$label, p = 0.8, list = FALSE)
train_data <- term_matrix[train_rows, ]
test_data <- term_matrix[-train_rows, ]
feature_names <- setdiff(names(term_matrix), c("document_id", "label"))

model <- naiveBayes(x = train_data[, feature_names], y = train_data$label)
predictions <- predict(model, test_data[, feature_names])
confusion <- confusionMatrix(predictions, test_data$label)

metrics <- data.frame(
  metric = c("samples", "features", "accuracy"),
  value = c(nrow(data), length(feature_names), unname(confusion$overall[["Accuracy"]]))
)
write_csv(metrics, file.path(output_dir, "metrics.csv"))
write.csv(as.data.frame(confusion$table), file.path(output_dir, "confusion_matrix.csv"), row.names = FALSE)

png(file.path(output_dir, "wordcloud.png"), width = 1200, height = 600)
set.seed(123)
frequencies <- tokens %>% count(word, sort = TRUE)
wordcloud(frequencies$word, frequencies$n, min.freq = 5, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png(file.path(output_dir, "top_words.png"), width = 1200, height = 700)
top_words <- frequencies %>% slice_max(n, n = 15)
print(ggplot(top_words, aes(x = reorder(word, n), y = n)) +
        geom_col(fill = "steelblue") +
        coord_flip() +
        labs(title = "Most frequent words", x = "Word", y = "Frequency") +
        theme_minimal())
dev.off()

cat("Analyzed", nrow(data), "tweets\n")
cat("Vocabulary size:", length(feature_names), "\n")
cat("Accuracy:", round(confusion$overall[["Accuracy"]], 4), "\n")
cat("Outputs saved to:", output_dir, "\n")
