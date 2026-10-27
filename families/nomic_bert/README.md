# Native tokenizer normalization

This family is implemented by the existing semantic `text_to_embedding` Task.
Its graph, runtime role prefixes, truncation and pooling remain defined by the
family builder and pipeline. This change only corrects native WordPiece
normalization and character classification.

The family-local `runtime/unicode_data.h` derives canonical decompositions,
nonspacing-mark ranges and simple lowercase mappings from fields 5, 2 and 13 of
[UnicodeData 8.0.0](https://www.unicode.org/Public/8.0.0/ucd/UnicodeData.txt),
SHA256 `38b17e1118206489a7e0ab5d29d7932212d38838df7d3ec025ecb58e8798ec20`.
The category version matches the Tokenizers 0.22.2 oracle used for these regressions;
case and decomposition mappings are pinned to Unicode 8.0.0.
Hangul decomposition is algorithmic. The Unicode notice is retained in
`UNICODE_LICENSE.txt`; no extra runtime dependency is introduced.

The existing native tokenizer test retains prefix/framing assertions and adds
distinct-vocabulary normalization cases, independent normalizer flags and
punctuation boundaries. Matching unknown tokens cannot stand in for the expected
IDs in those cases. These CPU checks do not establish model inference parity,
performance or exhaustive compatibility with newer Unicode versions.

Rebuild the family runtime to obtain corrected token IDs. No bundle schema or
public Task API change is required.
