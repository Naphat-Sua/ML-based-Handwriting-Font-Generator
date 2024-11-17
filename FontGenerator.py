FontGenerator = main()
if FontGenerator:
    ttf_path, metadata_path = FontGenerator(
        collected_samples,
        font_name="MyHandwriting",
        author="Your Name"
    )
