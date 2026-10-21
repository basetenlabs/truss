#![cfg(feature = "rustls")]

/// The rustls backend uses reqwest and Mozilla's bundled roots, not certifi.
/// Match both subject and key so a renamed anchor cannot reintroduce the G3 key.
#[test]
fn rustls_root_bundle_excludes_emsign_ecc_g3() {
    // emSign ECC Root CA - G3 certificate SHA-256:
    // 86:a1:ec:ba:08:9c:4a:8d:3b:be:27:34:c6:12:ba:34:
    // 1d:81:3e:04:3c:f9:e8:a8:62:cd:5c:57:a3:6b:be:6b
    const SUBJECT: &[u8] = b"1\x0b0\t\x06\x03U\x04\x06\x13\x02IN1\x130\x11\x06\x03U\x04\x0b\x13\nemSign PKI1%0#\x06\x03U\x04\n\x13\x1ceMudhra Technologies Limited1 0\x1e\x06\x03U\x04\x03\x13\x17emSign ECC Root CA - G3";
    const SPKI: &[u8] = b"0\x10\x06\x07*\x86H\xce=\x02\x01\x06\x05+\x81\x04\x00\"\x03b\x00\x04#\xa5\x0c\xb8-\x12\xf5(\xf3\xb1\xb2\xdd\xe2\x02\x12\x80\x9e9_IM\x9f\xc9%4Yt\xec\xbb\x06\x1c\xe7\xc0r\xaf\xe8\xae/\xe1AT\x87\x14\xa8J\xb2\xe8|\x82\xe6[j\xb5\xdc\xb3u\xce\x8b\x06\xd0\x86#\xbfF\xd5\x8e\x0f?\x04\xf4\xd7\x1c\x92~\xf6\xa5c\xc2\xf5_\x8e.O\xa1\x18\x19\x02+2\n\x82d}\x16\x93\xd1";

    let roots = webpki_roots::TLS_SERVER_ROOTS;
    assert!(!roots.is_empty(), "the TLS trust store must not be empty");
    for root in roots {
        assert_ne!(root.subject.as_ref(), SUBJECT, "emSign G3 is still trusted");
        assert_ne!(
            root.subject_public_key_info.as_ref(),
            SPKI,
            "emSign G3's public key is still trusted"
        );
    }
}
