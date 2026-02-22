resource "aws_s3_bucket" "data_lake" {
  bucket = "ai-at-scale-data-lake"
}

resource "aws_s3_bucket_lifecycle_configuration" "intelligent_tiering" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    id     = "intelligent_tiering_transition"
    status = "Enabled"

    transition {
      days          = 0
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}
