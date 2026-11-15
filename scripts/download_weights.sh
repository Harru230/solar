#!/usr/bin/env bash
# ==============================================================================
# download_weights.sh - Automated External FP32 Model Weights Fetch & Verification
#
# Downloads the 6 per-family FP32 checkpoint archives (< 820 MB each, 30 models total)
# from GitHub Releases into work/checkpoints/ for exact reproduction of Rules A & B,
# and verifies their SHA-256 checksums.
# (For offline demos and smoke tests, the 6 Fold-0 FP16 weights already live in weights/.)
#
# Usage:
#   bash scripts/download_weights.sh --all [--tag v1.0.0] [--repo <USER>/<REPO>]
#   bash scripts/download_weights.sh --family clahe [--tag v1.0.0] [--repo <USER>/<REPO>]
# ==============================================================================
set -euo pipefail

TAG="${SOLAR_WEIGHTS_TAG:-v1.0.0}"
REPO="${SOLAR_WEIGHTS_REPO:-Harru230/solar}"
TARGET_DIR="${SOLAR_WEIGHTS_DIR:-work/checkpoints}"
DOWNLOAD_ALL=1
FAMILIES=()
FORCE=0

VALID_FAMILIES=(clahe anchors5 dice 2048 a5dice 1792 refiner)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DOWNLOAD_ALL=1
            FAMILIES=()
            shift
            ;;
        --family)
            DOWNLOAD_ALL=0
            FAMILIES+=("$2")
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        --dir|--output)
            TARGET_DIR="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--all | --family <clahe|anchors5|dice|2048|a5dice|1792|refiner>] [--tag <tag>] [--repo <user/repo>] [--dir <output_dir>] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$DOWNLOAD_ALL" -eq 1 || "${#FAMILIES[@]}" -eq 0 ]]; then
    FAMILIES=("${VALID_FAMILIES[@]}")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_TARGET_DIR="${REPO_ROOT}/${TARGET_DIR}"
[[ -L "${FULL_TARGET_DIR}" ]] && rm -f "${FULL_TARGET_DIR}"
mkdir -p "${FULL_TARGET_DIR}"

declare -A FAMILY_FILES=(
    ["clahe"]="r50_1536_clahe_f0.pt r50_1536_clahe_f1.pt r50_1536_clahe_f2.pt r50_1536_clahe_f3.pt r50_1536_clahe_f4.pt"
    ["anchors5"]="r50_1536_anchors5_f0.pt r50_1536_anchors5_f1.pt r50_1536_anchors5_f2.pt r50_1536_anchors5_f3.pt r50_1536_anchors5_f4.pt"
    ["dice"]="r50_1536_dice_f0.pt r50_1536_dice_f1.pt r50_1536_dice_f2.pt r50_1536_dice_f3.pt r50_1536_dice_f4.pt"
    ["2048"]="r50_2048_f0.pt r50_2048_f1.pt r50_2048_f2.pt r50_2048_f3.pt r50_2048_f4.pt"
    ["a5dice"]="r50_1536_a5dice_f0.pt r50_1536_a5dice_f1.pt r50_1536_a5dice_f2.pt r50_1536_a5dice_f3.pt r50_1536_a5dice_f4.pt"
    ["1792"]="r50_1792_clahe_f0.pt r50_1792_clahe_f1.pt r50_1792_clahe_f2.pt r50_1792_clahe_f3.pt r50_1792_clahe_f4.pt"
    ["refiner"]="semantic_unet_f0_fp16.pt semantic_unet_f1_fp16.pt semantic_unet_f2_fp16.pt semantic_unet_f3_fp16.pt semantic_unet_f4_fp16.pt resnext101_unet_f0_fp16.pt resnext101_unet_f1_fp16.pt resnext101_unet_f2_fp16.pt resnext101_unet_f3_fp16.pt resnext101_unet_f4_fp16.pt crop_refiner_f0_fp16.pt crop_refiner_f1_fp16.pt crop_refiner_f2_fp16.pt crop_refiner_f3_fp16.pt crop_refiner_f4_fp16.pt wide_refiner_v2_f0_fp16.pt wide_refiner_v2_f1_fp16.pt wide_refiner_v2_f2_fp16.pt wide_refiner_v2_f3_fp16.pt wide_refiner_v2_f4_fp16.pt lgb_s27_yhyb_s2026_f0.txt lgb_s27_yhyb_s2026_f1.txt lgb_s27_yhyb_s2026_f2.txt lgb_s27_yhyb_s2026_f3.txt lgb_s27_yhyb_s2026_f4.txt lgb_s27_yhyb_s2027_f0.txt lgb_s27_yhyb_s2027_f1.txt lgb_s27_yhyb_s2027_f2.txt lgb_s27_yhyb_s2027_f3.txt lgb_s27_yhyb_s2027_f4.txt lgb_s27_yhyb_s2028_f0.txt lgb_s27_yhyb_s2028_f1.txt lgb_s27_yhyb_s2028_f2.txt lgb_s27_yhyb_s2028_f3.txt lgb_s27_yhyb_s2028_f4.txt lgb_s27_yhyb_s2029_f0.txt lgb_s27_yhyb_s2029_f1.txt lgb_s27_yhyb_s2029_f2.txt lgb_s27_yhyb_s2029_f3.txt lgb_s27_yhyb_s2029_f4.txt lgb_s27_yhyb_s2030_f0.txt lgb_s27_yhyb_s2030_f1.txt lgb_s27_yhyb_s2030_f2.txt lgb_s27_yhyb_s2030_f3.txt lgb_s27_yhyb_s2030_f4.txt lgb_metadata.json"
)

declare -A KNOWN_CHECKSUMS=(
    ["r50_1536_clahe_f0.pt"]="060fdd76ebdef93cdc99f7cc7d68285f08109a35687b7043fe4d8b885ac0e87f"
    ["r50_1536_clahe_f1.pt"]="2729cee16f340ab18cf56695398db1ba3d84d02e94bb8982e3295bc28b533133"
    ["r50_1536_clahe_f2.pt"]="e0bed6b916b4d5db2cc0b059aee3890166b66e2fc3e87320cc1dd190c050226a"
    ["r50_1536_clahe_f3.pt"]="27827a45d5d3adeb43f30f8e454a4cc3ef78b7b6f4d1e7d90492d9811b39b72d"
    ["r50_1536_clahe_f4.pt"]="771d050d3267d2d4cc23a514eb18cf8f9ecc7d6287f165e14e8d214ea5d587cc"
    ["r50_1536_anchors5_f0.pt"]="97bbf20e3ce92e411cd3b90e5afed51b06f2d4e9796201a4124d1bc412a2564f"
    ["r50_1536_anchors5_f1.pt"]="3efda476597ac106e96fcdca7382609dd8fa567682bfd1bd54da9c46c77a6fa9"
    ["r50_1536_anchors5_f2.pt"]="6850953e83a6df5cf1d4c994ddf70cd09fca508c9fb9b465c6fc33e6eb91c643"
    ["r50_1536_anchors5_f3.pt"]="fd5c26ed390dabdcdf009fc58652f85b05347f6c8c92e742e6ab42a6c64a45fe"
    ["r50_1536_anchors5_f4.pt"]="1dc8e42631f66a3ae8d3831d8331cc65411794bec52f28fa489e4dbb4baa95aa"
    ["r50_1536_dice_f0.pt"]="5f1b3f22d193998b2befc8185f2688e0329663f0e53015e410ac886867d1b610"
    ["r50_1536_dice_f1.pt"]="508b6c85a9976b5ed943d81d39d9c1d26c122f465e19a6bb7933bc3e95b12d51"
    ["r50_1536_dice_f2.pt"]="91a52f4a259761635dc349ae28a6ce7bdc170366c457f6807a5a40f78f9f1512"
    ["r50_1536_dice_f3.pt"]="25fd20dd6c52fedd33ff37f174541602556dff95dea37262bf035acec82829cf"
    ["r50_1536_dice_f4.pt"]="c597cfe41308aa5786b6172380afd57367b3c7a86fb468a851a66f71f4dbca7b"
    ["r50_2048_f0.pt"]="fff7989cb52ab751ac40613ef1a4b14199b3b2bc35a377212b01e90c8078d78a"
    ["r50_2048_f1.pt"]="b41388b85712b7f0cef9aa3a6ac52546335a0d269cab65d02ae57d4017fc3a42"
    ["r50_2048_f2.pt"]="e9da2ae20fefe392cb190ec150729a896dcff97acfe00fe421488ce61a237005"
    ["r50_2048_f3.pt"]="01913a10f5d224049a04f6815d633dc0adffff315dd7a0a5f448426e32b47fb8"
    ["r50_2048_f4.pt"]="1380b400ea16014c928baa4f93f1fa609c82886a46a351779d8e496ade82e1b1"
    ["r50_1536_a5dice_f0.pt"]="5b6256933aae7d7aec5996c04d08545575c40160fe2e0f48b77990fbaea76256"
    ["r50_1536_a5dice_f1.pt"]="7c992a4ea280e559ed91c617048f2a847b25e1cdc8e45cd73cdb21a85e3d4532"
    ["r50_1536_a5dice_f2.pt"]="f5df4b386ddd641c0093c2aae7c02ecc3779b807a9a4a9ac7859cdc20e5449d4"
    ["r50_1536_a5dice_f3.pt"]="d9b617e2faff64be2cd483468c825c5cc70e12f07036206844c8873225d7b60d"
    ["r50_1536_a5dice_f4.pt"]="5ef70f009889c276fc0635f85da71efdaa24c0780cc838ddcbd9b2adfb6b3dd0"
    ["r50_1792_clahe_f0.pt"]="f8f5161311916f89729a1b351d8df96f6c2c1de90d668a1e121d55b72ecea923"
    ["r50_1792_clahe_f1.pt"]="552a2cb263cae3810204423a5ba0a5ff08797341a30a63c805e7d3c7dc847836"
    ["r50_1792_clahe_f2.pt"]="ee915ec97105bf50e7fdabf2897d7bbeedf303a1c6a2f4abc7b66e59b908c1c2"
    ["r50_1792_clahe_f3.pt"]="ade1836dd80645762e38bd0685b17fb406d80b59dad3874eeb4266f0c2ed7509"
    ["r50_1792_clahe_f4.pt"]="c0712263ac8c08d773a0f244f1a753c87871ee280010186c880fdf7c65dc462f"
    ["semantic_unet_f0_fp16.pt"]="c7633815a7b72bb5349837411b410b50318730e47624b3e5f1a589f0afd3bd4b"
    ["semantic_unet_f1_fp16.pt"]="2f64765d38cb5aa8b30d99fc9dd5bcbe1d7182e2a12302b355124af0b3947ed2"
    ["semantic_unet_f2_fp16.pt"]="3eb6523e5311cf7539fbf8f3fe3b7e21dd91939c289c9344062504cc3ffdc590"
    ["semantic_unet_f3_fp16.pt"]="70d91bf3a57a4113b4865672520e168ce06e02a47abf0b2d91eb6eed33847cb1"
    ["semantic_unet_f4_fp16.pt"]="a2fcfa137fffa2ecc8baa7fcb7a352bec3f0889ed2eb1ff7b2b02aeb5bf48240"
    ["resnext101_unet_f0_fp16.pt"]="87b1dd8eaf667fe89c2de85ade03cf740bbdb480e8b1ec64db2458390a25ed1f"
    ["resnext101_unet_f1_fp16.pt"]="768e2708cc2647f5598a4edc79ffdd6d4c3b57a11d3ca6f518d426f29a5f1891"
    ["resnext101_unet_f2_fp16.pt"]="7f0ec41071f7076b7157baac0dfdb7cdd121cb30c6935b10b3693ac1b11dda34"
    ["resnext101_unet_f3_fp16.pt"]="8bf01878bc55d7ab9b3fcb2689fe823ffc0615740734e81fbd41f06a206b720e"
    ["resnext101_unet_f4_fp16.pt"]="86860868e54bad3dc6737f7ef52794c1a949f728df8b3489165a689c39157e77"
    ["crop_refiner_f0_fp16.pt"]="fe1a75f695450bbbdbbd15f34c505fc316133f5d17276fd895c314ca88810b88"
    ["crop_refiner_f1_fp16.pt"]="a6a45ce16f27c5e07bb0d1710e9a50fc58ad56965ccf09cccf65ef60ff765395"
    ["crop_refiner_f2_fp16.pt"]="8de9272517b164f30bbce5e3ddc4bfc26968f4f0a85b609493f37599a6b0bfa0"
    ["crop_refiner_f3_fp16.pt"]="e82ba6553796f85f76f20ee87a901a16d3d2cf58f1c708803e669e0dd547fea4"
    ["crop_refiner_f4_fp16.pt"]="414dbed8655e3638b5431ec707b24c034ddb200e339e0cd85d0b592df47bc86f"
    ["wide_refiner_v2_f0_fp16.pt"]="f753a39aa588cc90dd8d23514d901f1b014c676614f66c4a1fff0886e7ee9132"
    ["wide_refiner_v2_f1_fp16.pt"]="700c5b4fb22cbd7d0afb66df0e642bd5b94b3569dca9265acfe88f294e9de549"
    ["wide_refiner_v2_f2_fp16.pt"]="7de92cb75043f8408d1a197daac175b1e61d1f67eb6e3b0f21bdca27032ecefc"
    ["wide_refiner_v2_f3_fp16.pt"]="a9eea775d298b9dbe39166447206aa3d2f78e92f136e937dceb7a48da626429a"
    ["wide_refiner_v2_f4_fp16.pt"]="0d10a86dfa7c39f4451a8dd24c0ae2b044009a305b8b5a1a9bfd764e3a348446"
    ["lgb_s27_yhyb_s2026_f0.txt"]="dfe8c7bce79fdd7b912d8729555907ecf5fa91489853ab14e46cae4fbde73d3c"
    ["lgb_s27_yhyb_s2026_f1.txt"]="b6a3e9e45fdfd91e66a2bf9513a1e03b0f0746531cba72077872b9b067b622e7"
    ["lgb_s27_yhyb_s2026_f2.txt"]="57a7ac8486e6b791fd14b9d01da5226e20eaad8fb3e9c43c9aa333b4910708b5"
    ["lgb_s27_yhyb_s2026_f3.txt"]="ebe109c27d3480704c16afe27f5c985daaab2b87cbfd95f2b41809bc211422ef"
    ["lgb_s27_yhyb_s2026_f4.txt"]="4f69b18579e8bb3e5ca94e740499a711476037cf0b4d971994ac8360a91c1e6e"
    ["lgb_s27_yhyb_s2027_f0.txt"]="28c0282bba55ef1743aabfa4d52aca895cffbf954c47b6005f459e6a706abd7e"
    ["lgb_s27_yhyb_s2027_f1.txt"]="894297fbb96053fd260053d31c97d3fbaac49c0df2c8ef5aa4d93605822b4601"
    ["lgb_s27_yhyb_s2027_f2.txt"]="a09cd5ef2b5b3ec8756dbb47fccd269721c08fa9dda2cd476c88cf1d394e683c"
    ["lgb_s27_yhyb_s2027_f3.txt"]="d2951cdbd1ffaa9d40fadec9dfeb7fc99dd0115cdf1008fae20b0317524b0261"
    ["lgb_s27_yhyb_s2027_f4.txt"]="33e94cf228e19ad0948dfbdaa23e04465c1de85b4204673eb1baa771b32e149a"
    ["lgb_s27_yhyb_s2028_f0.txt"]="370b5995ee0b4a2ac26a995b9f08633abf5473f7a8f8ac0cb2d6d79c08363bf7"
    ["lgb_s27_yhyb_s2028_f1.txt"]="670d3034576d613a141a38f3636edf0c686b69ad8c3b68a1b4a7401e5455a2aa"
    ["lgb_s27_yhyb_s2028_f2.txt"]="aef567b108ce091ee3f70af57c94f5d19c2e55bf2ef83f08d054053b70ea48b7"
    ["lgb_s27_yhyb_s2028_f3.txt"]="9c9d2fd2439abd70d3850e3475faa199cfc1b2f9788c27c5369c31ef091841a1"
    ["lgb_s27_yhyb_s2028_f4.txt"]="4e639d1ab28ea5f12454723ae2b38bd2412bb5f371655fb12ee15961cf6d806a"
    ["lgb_s27_yhyb_s2029_f0.txt"]="9c3b9d527c0980001fbf6fb1d4af884e4b5d1bd0117c51f687858e37148d8a6f"
    ["lgb_s27_yhyb_s2029_f1.txt"]="2decfa2ead1cfc419b0ca4689a9ff50eb32c0bf7e21090c6b4394396813919e1"
    ["lgb_s27_yhyb_s2029_f2.txt"]="0e9fca77b5eef1855df7bc19de4b4a6893ca1fed5a23b41cc1fc30c7878a258b"
    ["lgb_s27_yhyb_s2029_f3.txt"]="712084591b780187f0e22166e71c9ca7bfd1dea96b0512f8dfa94ec272720902"
    ["lgb_s27_yhyb_s2029_f4.txt"]="862468e7537fe647d9c0f0245c47189c0587762d84d3536819fc4cc15915ac49"
    ["lgb_s27_yhyb_s2030_f0.txt"]="68918ccd3e298d71035b32e48f0f96946aaa1e760bd4f4112018f1504c190f42"
    ["lgb_s27_yhyb_s2030_f1.txt"]="7e9687491ada81ddb462f5bd13f2508d1d395c04aae6af3711a38a6dfb4dc883"
    ["lgb_s27_yhyb_s2030_f2.txt"]="d603ce53f383dabf9ff08b664c567592dd4bec77b7007e15359a1ac437dbc38e"
    ["lgb_s27_yhyb_s2030_f3.txt"]="547f336c2864ec10de48f1e63620a67a2251097d8ded10b78b1bf8f2712dc411"
    ["lgb_s27_yhyb_s2030_f4.txt"]="5bb7504a6f55db18ceb99bcc43c85ec29007509a988d13b2c81b878264fc66ee"
    ["lgb_metadata.json"]="2fa3a844b8f81dcc99e7d0a464a4163b470c3b25c9ce27d7342a1fc5eba1e6d5"
)

verify_family() {
    local fam="$1"
    for fname in ${FAMILY_FILES[$fam]}; do
        local fpath="${FULL_TARGET_DIR}/${fname}"
        [[ -f "$fpath" ]] || return 1
        local actual
        actual="$(sha256sum "$fpath" | awk '{print $1}')"
        if [[ "$actual" != "${KNOWN_CHECKSUMS[$fname]}" ]]; then
            echo "[ERROR] SHA-256 mismatch for ${fname}: expected ${KNOWN_CHECKSUMS[$fname]}, got ${actual}"
            return 1
        fi
        echo "${fname}: OK"
    done
    return 0
}

download_archive() {
    local archive="$1"
    local url="https://github.com/${REPO}/releases/download/${TAG}/${archive}"
    local dest="${FULL_TARGET_DIR}/${archive}"
    if command -v gh &>/dev/null; then
        if gh release download "$TAG" --repo "$REPO" --pattern "$archive" --dir "$FULL_TARGET_DIR" --clobber; then
            return 0
        fi
    fi
    if command -v curl &>/dev/null; then
        if curl -fL --progress-bar "$url" -o "$dest"; then
            return 0
        fi
    fi
    if command -v wget &>/dev/null; then
        if wget -q --show-progress "$url" -O "$dest"; then
            return 0
        fi
    fi
    return 1
}

for fam in "${FAMILIES[@]}"; do
    if [[ -z "${FAMILY_FILES[$fam]:-}" ]]; then
        echo "[ERROR] Unknown family '${fam}'. Valid options: ${VALID_FAMILIES[*]}"
        exit 1
    fi
    if [[ "$FORCE" -eq 0 ]] && verify_family "$fam" 2>/dev/null; then
        echo "[OK] Family '${fam}' already present and verified in ${FULL_TARGET_DIR}."
        continue
    fi
    archive="weights_family_${fam}.tar.gz"
    echo "==> Downloading ${archive} from ${REPO} release ${TAG}..."
    if ! download_archive "$archive"; then
        echo "[ERROR] Failed to fetch ${archive} (release '${TAG}' on '${REPO}' may be private or not yet published)."
        exit 1
    fi
    tar -xzf "${FULL_TARGET_DIR}/${archive}" -C "$FULL_TARGET_DIR"
    rm -f "${FULL_TARGET_DIR}/${archive}"
    verify_family "$fam"
done

echo "[SUCCESS] Requested FP32 weights verified in ${FULL_TARGET_DIR}."
