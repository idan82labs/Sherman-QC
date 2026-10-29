#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#pragma STDC FENV_ACCESS ON

#ifndef CIIP_SOURCE_SHA256
#define CIIP_SOURCE_SHA256 "UNBOUND"
#endif

#ifndef CIIP_BUILD_FLAGS
#define CIIP_BUILD_FLAGS "UNBOUND"
#endif

namespace {

using i128 = __int128;
using u128 = unsigned __int128;

constexpr int kOk = 0;
constexpr int kNull = 1;
constexpr int kRoundingMode = 2;
constexpr int kNonfinite = 3;
constexpr int kOverflow = 4;
constexpr int kResidual = 5;
constexpr int kNegativeEigenvalue = 6;
constexpr int kDigestCollision = 7;

class Sha256 {
 public:
  Sha256() noexcept
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(const void* source, std::size_t length) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(source);
    bit_length_ += static_cast<std::uint64_t>(length) * 8U;
    while (length != 0) {
      const std::size_t take = std::min<std::size_t>(length, 64U - buffer_length_);
      std::memcpy(buffer_.data() + buffer_length_, bytes, take);
      buffer_length_ += take;
      bytes += take;
      length -= take;
      if (buffer_length_ == 64U) {
        transform(buffer_.data());
        buffer_length_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() noexcept {
    const std::uint64_t original_bit_length = bit_length_;
    const std::uint8_t one = 0x80U;
    update(&one, 1);
    const std::uint8_t zero = 0;
    while (buffer_length_ != 56U) update(&zero, 1);
    std::uint8_t length_bytes[8];
    for (int index = 0; index < 8; ++index) {
      length_bytes[7 - index] = static_cast<std::uint8_t>(original_bit_length >> (8 * index));
    }
    update(length_bytes, 8);
    std::array<std::uint8_t, 32> output{};
    for (int index = 0; index < 8; ++index) {
      output[4 * index] = static_cast<std::uint8_t>(state_[index] >> 24);
      output[4 * index + 1] = static_cast<std::uint8_t>(state_[index] >> 16);
      output[4 * index + 2] = static_cast<std::uint8_t>(state_[index] >> 8);
      output[4 * index + 3] = static_cast<std::uint8_t>(state_[index]);
    }
    return output;
  }

 private:
  static std::uint32_t rotr(std::uint32_t value, int count) noexcept {
    return (value >> count) | (value << (32 - count));
  }

  void transform(const std::uint8_t block[64]) noexcept {
    static constexpr std::uint32_t constants[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
    };
    std::uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
      words[index] = (static_cast<std::uint32_t>(block[4 * index]) << 24) |
                     (static_cast<std::uint32_t>(block[4 * index + 1]) << 16) |
                     (static_cast<std::uint32_t>(block[4 * index + 2]) << 8) |
                     static_cast<std::uint32_t>(block[4 * index + 3]);
    }
    for (int index = 16; index < 64; ++index) {
      const std::uint32_t s0 = rotr(words[index - 15], 7) ^ rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const std::uint32_t s1 = rotr(words[index - 2], 17) ^ rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
    std::uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
    for (int index = 0; index < 64; ++index) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
    state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_length_ = 0;
  std::uint64_t bit_length_ = 0;
};

template <typename Integer>
void sha_update_le(Sha256* sha, Integer value) noexcept {
  std::uint8_t bytes[sizeof(Integer)];
  using Unsigned = typename std::make_unsigned<Integer>::type;
  const Unsigned converted = static_cast<Unsigned>(value);
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    bytes[index] = static_cast<std::uint8_t>(converted >> (8 * index));
  }
  sha->update(bytes, sizeof(bytes));
}

Sha256 tagged_sha(const char* tag) noexcept {
  Sha256 sha;
  sha.update(tag, std::strlen(tag) + 1);
  return sha;
}

struct I128Parts {
  std::uint64_t lo;
  std::int64_t hi;
};

struct U128Parts {
  std::uint64_t lo;
  std::uint64_t hi;
};

struct JacobiOutput {
  double lambda1;
  double lambda2;
  double lambda3;
  double trace;
  double max_offdiag;
};

struct AnisotropyOutput {
  double scalar;
  std::uint16_t q;
  std::uint16_t filtration;
};

inline i128 from_parts(I128Parts value) noexcept {
  const u128 bits = (static_cast<u128>(static_cast<std::uint64_t>(value.hi)) << 64) |
                    static_cast<u128>(value.lo);
  return static_cast<i128>(bits);
}

inline I128Parts to_parts(i128 value) noexcept {
  const u128 bits = static_cast<u128>(value);
  return I128Parts{static_cast<std::uint64_t>(bits),
                   static_cast<std::int64_t>(static_cast<std::uint64_t>(bits >> 64))};
}

inline U128Parts to_parts(u128 value) noexcept {
  return U128Parts{static_cast<std::uint64_t>(value), static_cast<std::uint64_t>(value >> 64)};
}

inline bool rounding_is_locked() noexcept { return std::fegetround() == FE_TONEAREST; }

// Volatile stores force one binary64 result per listed primitive. The build also
// disables contraction and every fast/unsafe-math transformation.
inline double rn_add(double left, double right) noexcept {
  volatile double result = left + right;
  return result;
}

inline double rn_sub(double left, double right) noexcept {
  volatile double result = left - right;
  return result;
}

inline double rn_mul(double left, double right) noexcept {
  volatile double result = left * right;
  return result;
}

inline double rn_div(double left, double right) noexcept {
  volatile double result = left / right;
  return result;
}

inline double rn_sqrt(double value) noexcept {
  volatile double result = std::sqrt(value);
  return result;
}

inline bool finite(double value) noexcept { return std::isfinite(value); }

bool checked_pow4(std::uint64_t value, u128* out) noexcept {
  u128 square = 0;
  const u128 wide = static_cast<u128>(value);
  if (__builtin_mul_overflow(wide, wide, &square)) return false;
  return !__builtin_mul_overflow(square, square, out);
}

bool scaled(u128 value, std::uint64_t multiplier, u128* out) noexcept {
  return !__builtin_mul_overflow(value, static_cast<u128>(multiplier), out);
}

inline std::uint32_t morton5(std::uint8_t x, std::uint8_t y, std::uint8_t z) noexcept {
  std::uint32_t result = 0;
  for (int bit = 0; bit < 5; ++bit) {
    result |= static_cast<std::uint32_t>((x >> bit) & 1U) << (3 * bit);
    result |= static_cast<std::uint32_t>((y >> bit) & 1U) << (3 * bit + 1);
    result |= static_cast<std::uint32_t>((z >> bit) & 1U) << (3 * bit + 2);
  }
  return result;
}

bool accumulate_i128(i128* accumulator, i128 value) noexcept {
  i128 result = 0;
  if (__builtin_add_overflow(*accumulator, value, &result)) return false;
  *accumulator = result;
  return true;
}

bool emit_i64(i128 value, std::int64_t* output) noexcept {
  if (value < static_cast<i128>(std::numeric_limits<std::int64_t>::min()) ||
      value > static_cast<i128>(std::numeric_limits<std::int64_t>::max())) return false;
  *output = static_cast<std::int64_t>(value);
  return true;
}

void matrix_multiply(const double left[3][3], const double right[3][3], double out[3][3]) noexcept {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double accumulator = 0.0;
      for (int k = 0; k < 3; ++k) {
        accumulator = rn_add(accumulator, rn_mul(left[i][k], right[k][j]));
      }
      out[i][j] = accumulator;
    }
  }
}

bool rotate(double matrix[3][3], int p, int q) noexcept {
  const double apq = matrix[p][q];
  if (apq == 0.0) return true;

  const double numerator = rn_sub(matrix[q][q], matrix[p][p]);
  const double denominator = rn_mul(2.0, apq);
  const double tau = rn_div(numerator, denominator);
  if (!finite(tau)) return false;

  double t = 1.0;
  if (tau != 0.0) {
    const double tau_squared = rn_mul(tau, tau);
    const double root_argument = rn_add(1.0, tau_squared);
    const double root = rn_sqrt(root_argument);
    const double t_denominator = rn_add(std::fabs(tau), root);
    const double sign = std::signbit(tau) ? -1.0 : 1.0;
    t = rn_div(sign, t_denominator);
  }
  const double t_squared = rn_mul(t, t);
  const double c = rn_div(1.0, rn_sqrt(rn_add(1.0, t_squared)));
  const double s = rn_mul(t, c);
  if (!finite(t) || !finite(c) || !finite(s)) return false;

  double rotation[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  rotation[p][p] = c;
  rotation[q][q] = c;
  rotation[p][q] = s;
  rotation[q][p] = -s;

  double first[3][3];
  matrix_multiply(matrix, rotation, first);
  double transpose[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) transpose[i][j] = rotation[j][i];
  }
  double updated[3][3];
  matrix_multiply(transpose, first, updated);
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      matrix[i][j] = updated[i][j];
      matrix[j][i] = matrix[i][j];
    }
  }
  return true;
}

#pragma pack(push, 1)
struct EdgeRow {
  std::uint8_t edge_id[16];
  std::uint32_t lower;
  std::uint32_t upper;
  std::uint8_t kind;
  std::uint8_t axis;
  std::uint16_t status;
  std::uint32_t reserved;
};

struct FieldRow {
  std::uint8_t row_id[16];
  std::uint8_t support_digest[16];
  std::uint64_t morton;
  std::int64_t trace_q;
  std::int64_t lambda1_q;
  std::int64_t lambda2_q;
  std::int64_t lambda3_q;
  std::int64_t mass_q;
  std::uint8_t arm;
  std::uint8_t scale;
  std::uint16_t status;
  std::uint32_t support_count;
  std::uint32_t raw_p10;
  std::uint32_t raw_p90;
  std::uint32_t balance_numerator;
  std::uint32_t balance_denominator;
  std::int32_t growth12;
  std::int32_t growth23;
  std::uint16_t anisotropy_q;
  std::uint16_t filtration;
  std::uint32_t stage_mask;
  std::uint16_t first_rejection;
  std::uint16_t flags;
  std::uint32_t eigenvalue_denominator;
};

struct PersistenceRow {
  std::uint8_t interval_id[16];
  std::uint8_t birth_id[16];
  std::uint8_t death_id[16];
  std::uint8_t arm;
  std::uint8_t status;
  std::uint16_t reason;
  std::uint16_t birth_f;
  std::uint16_t death_f;
  std::uint16_t lifetime;
  std::uint8_t scale_mask;
  std::uint32_t spatial_count;
  std::uint8_t flags;
};
#pragma pack(pop)

static_assert(sizeof(EdgeRow) == 32, "edge row width drift");
static_assert(sizeof(FieldRow) == 128, "field row width drift");
static_assert(sizeof(PersistenceRow) == 64, "persistence row width drift");

void copy_first16(Sha256* sha, std::uint8_t output[16]) noexcept {
  const auto digest = sha->finish();
  std::memcpy(output, digest.data(), 16);
}

bool digest_matches_hex(const std::uint8_t* digest, std::size_t length, const char* expected) noexcept {
  for (std::size_t index = 0; index < length; ++index) {
    const auto nibble = [](char value) -> int {
      if (value >= '0' && value <= '9') return value - '0';
      if (value >= 'a' && value <= 'f') return value - 'a' + 10;
      return -1;
    };
    const int high = nibble(expected[2 * index]);
    const int low = nibble(expected[2 * index + 1]);
    if (high < 0 || low < 0 || digest[index] != static_cast<std::uint8_t>((high << 4) | low)) return false;
  }
  return expected[2 * length] == '\0';
}

void hash_edge(const std::uint8_t case_digest[32], std::uint32_t lower, std::uint32_t upper,
               std::uint8_t kind, std::uint8_t axis, std::uint8_t output[16]) noexcept {
  Sha256 sha = tagged_sha("CIIP_EDGE_V1");
  sha.update(case_digest, 32);
  sha_update_le(&sha, lower);
  sha_update_le(&sha, upper);
  sha_update_le(&sha, kind);
  sha_update_le(&sha, axis);
  copy_first16(&sha, output);
}

void hash_field(const std::uint8_t case_digest[32], std::uint8_t arm, std::uint8_t scale,
                std::uint64_t morton, std::uint8_t output[16]) noexcept {
  Sha256 sha = tagged_sha("CIIP_FIELD_ROW_V1");
  sha.update(case_digest, 32);
  sha_update_le(&sha, arm);
  sha_update_le(&sha, scale);
  sha_update_le(&sha, morton);
  copy_first16(&sha, output);
}

void hash_essential(const std::uint8_t case_digest[32], std::uint8_t arm,
                    const std::uint8_t birth_id[16], std::uint8_t output[16]) noexcept {
  Sha256 sha = tagged_sha("CIIP_ESSENTIAL_DEATH_V1");
  sha.update(case_digest, 32);
  sha_update_le(&sha, arm);
  sha.update(birth_id, 16);
  copy_first16(&sha, output);
}

void hash_interval(const std::uint8_t case_digest[32], std::uint8_t arm,
                   const std::uint8_t birth_id[16], const std::uint8_t death_id[16],
                   std::uint16_t birth_f, std::uint16_t death_f,
                   std::uint8_t output[16]) noexcept {
  Sha256 sha = tagged_sha("CIIP_INTERVAL_V1");
  sha.update(case_digest, 32);
  sha_update_le(&sha, arm);
  sha.update(birth_id, 16);
  sha.update(death_id, 16);
  sha_update_le(&sha, birth_f);
  sha_update_le(&sha, death_f);
  copy_first16(&sha, output);
}

struct EdgeCollision {
  std::array<std::uint8_t, 16> digest;
  std::uint32_t lower;
  std::uint32_t upper;
  std::uint8_t kind;
  std::uint8_t axis;
};

bool edge_collision_free(const std::vector<EdgeRow>& rows) {
  std::vector<EdgeCollision> entries(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    std::memcpy(entries[index].digest.data(), rows[index].edge_id, 16);
    entries[index].lower = rows[index].lower;
    entries[index].upper = rows[index].upper;
    entries[index].kind = rows[index].kind;
    entries[index].axis = rows[index].axis;
  }
  std::sort(entries.begin(), entries.end(), [](const EdgeCollision& left, const EdgeCollision& right) {
    const int digest_order = std::memcmp(left.digest.data(), right.digest.data(), 16);
    if (digest_order != 0) return digest_order < 0;
    if (left.kind != right.kind) return left.kind < right.kind;
    if (left.axis != right.axis) return left.axis < right.axis;
    if (left.lower != right.lower) return left.lower < right.lower;
    return left.upper < right.upper;
  });
  for (std::size_t index = 1; index < entries.size(); ++index) {
    if (entries[index - 1].digest == entries[index].digest &&
        (entries[index - 1].lower != entries[index].lower ||
         entries[index - 1].upper != entries[index].upper ||
         entries[index - 1].kind != entries[index].kind ||
         entries[index - 1].axis != entries[index].axis)) return false;
  }
  return true;
}

int build_adjacency_rows(const std::uint8_t case_digest[32], const std::uint64_t* mortons,
                         std::uint32_t occupied_count, std::vector<EdgeRow>* output) {
  if (case_digest == nullptr || mortons == nullptr || output == nullptr || occupied_count == 0 ||
      occupied_count > 32768) return kNull;
  std::array<std::int32_t, 32768> ordinal{};
  ordinal.fill(-1);
  for (std::uint32_t index = 0; index < occupied_count; ++index) {
    if (mortons[index] >= (1U << 15) || (index != 0 && mortons[index] <= mortons[index - 1])) return kOverflow;
    ordinal[static_cast<std::size_t>(mortons[index])] = static_cast<std::int32_t>(index);
  }
  output->clear();
  output->reserve(static_cast<std::size_t>(11) * occupied_count);
  for (std::uint8_t axis = 0; axis < 3; ++axis) {
    for (std::uint32_t source = 0; source < occupied_count; ++source) {
      std::uint8_t x = 0, y = 0, z = 0;
      const std::uint64_t morton = mortons[source];
      for (int bit = 0; bit < 5; ++bit) {
        x |= static_cast<std::uint8_t>(((morton >> (3 * bit)) & 1U) << bit);
        y |= static_cast<std::uint8_t>(((morton >> (3 * bit + 1)) & 1U) << bit);
        z |= static_cast<std::uint8_t>(((morton >> (3 * bit + 2)) & 1U) << bit);
      }
      std::uint8_t* coordinate = axis == 0 ? &x : (axis == 1 ? &y : &z);
      if (*coordinate >= 31) continue;
      ++(*coordinate);
      const std::int32_t neighbor = ordinal[morton5(x, y, z)];
      if (neighbor < 0) continue;
      for (std::uint8_t scale = 0; scale < 3; ++scale) {
        const std::uint32_t left = source + static_cast<std::uint32_t>(scale) * occupied_count;
        const std::uint32_t right = static_cast<std::uint32_t>(neighbor) +
                                    static_cast<std::uint32_t>(scale) * occupied_count;
        EdgeRow row{};
        row.lower = std::min(left, right);
        row.upper = std::max(left, right);
        row.kind = 0;
        row.axis = axis;
        output->push_back(row);
      }
    }
  }
  for (std::uint8_t scale = 0; scale < 2; ++scale) {
    for (std::uint32_t ordinal_index = 0; ordinal_index < occupied_count; ++ordinal_index) {
      EdgeRow row{};
      row.lower = ordinal_index + static_cast<std::uint32_t>(scale) * occupied_count;
      row.upper = ordinal_index + static_cast<std::uint32_t>(scale + 1) * occupied_count;
      row.kind = 1;
      row.axis = 255;
      output->push_back(row);
    }
  }
  std::sort(output->begin(), output->end(), [](const EdgeRow& left, const EdgeRow& right) {
    if (left.kind != right.kind) return left.kind < right.kind;
    if (left.axis != right.axis) return left.axis < right.axis;
    if (left.lower != right.lower) return left.lower < right.lower;
    return left.upper < right.upper;
  });
  if (output->size() < static_cast<std::size_t>(2) * occupied_count ||
      output->size() > static_cast<std::size_t>(11) * occupied_count) return kOverflow;
  for (auto& row : *output) hash_edge(case_digest, row.lower, row.upper, row.kind, row.axis, row.edge_id);
  return edge_collision_free(*output) ? kOk : kDigestCollision;
}

}  // namespace

extern "C" {

const char* ciip_v1_source_sha256() noexcept { return CIIP_SOURCE_SHA256; }
const char* ciip_v1_build_flags() noexcept { return CIIP_BUILD_FLAGS; }
const char* ciip_v1_compiler_version() noexcept { return __clang_version__; }
int ciip_v1_status_ok() noexcept { return kOk; }

int ciip_v1_i128_add(I128Parts left, I128Parts right, I128Parts* out) noexcept {
  if (out == nullptr) return kNull;
  i128 result = 0;
  if (__builtin_add_overflow(from_parts(left), from_parts(right), &result)) return kOverflow;
  *out = to_parts(result);
  return kOk;
}

int ciip_v1_i128_accumulate_product(
    I128Parts accumulator, std::int64_t left, std::int64_t right, I128Parts* out) noexcept {
  if (out == nullptr) return kNull;
  i128 product = 0;
  if (__builtin_mul_overflow(static_cast<i128>(left), static_cast<i128>(right), &product)) return kOverflow;
  i128 result = 0;
  if (__builtin_add_overflow(from_parts(accumulator), product, &result)) return kOverflow;
  *out = to_parts(result);
  return kOk;
}

int ciip_v1_u128_pow4(std::uint64_t value, U128Parts* out) noexcept {
  if (out == nullptr) return kNull;
  u128 result = 0;
  if (!checked_pow4(value, &result)) return kOverflow;
  *out = to_parts(result);
  return kOk;
}

int ciip_v1_growth_gate(
    std::uint64_t n1, std::uint64_t n2, std::uint64_t n3,
    std::int32_t* growth12_code, std::int32_t* growth23_code) noexcept {
  if (growth12_code == nullptr || growth23_code == nullptr) return kNull;
  u128 p1 = 0, p2 = 0, p3 = 0;
  if (!checked_pow4(n1, &p1) || !checked_pow4(n2, &p2) || !checked_pow4(n3, &p3)) return kOverflow;
  u128 left = 0, right = 0;
  if (!scaled(p2, 32, &left) || !scaled(p1, 243, &right)) return kOverflow;
  if (left < right) {
    *growth12_code = -1;
  } else {
    if (!scaled(p2, 2048, &left) || !scaled(p1, 177147, &right)) return kOverflow;
    *growth12_code = left > right ? 1 : 0;
  }
  if (!scaled(p3, 243, &left) || !scaled(p2, 1024, &right)) return kOverflow;
  if (left < right) {
    *growth23_code = -1;
  } else {
    if (!scaled(p3, 177147, &left) || !scaled(p2, 4194304, &right)) return kOverflow;
    *growth23_code = left > right ? 1 : 0;
  }
  return kOk;
}

int ciip_v1_density_gate(std::uint64_t p90, std::uint64_t p10, std::uint8_t* passes) noexcept {
  if (passes == nullptr) return kNull;
  const std::uint64_t denominator = std::max<std::uint64_t>(1, p10);
  u128 limit = 0;
  if (__builtin_mul_overflow(static_cast<u128>(32), static_cast<u128>(denominator), &limit)) return kOverflow;
  *passes = static_cast<std::uint8_t>(static_cast<u128>(p90) <= limit);
  return kOk;
}

int ciip_v1_balance_gate(std::uint64_t numerator, std::uint64_t denominator, std::uint8_t* passes) noexcept {
  if (passes == nullptr) return kNull;
  if (denominator == 0 || numerator > denominator) return kOverflow;
  u128 left = 0, right = 0;
  if (__builtin_mul_overflow(static_cast<u128>(20), static_cast<u128>(numerator), &left) ||
      __builtin_mul_overflow(static_cast<u128>(11), static_cast<u128>(denominator), &right)) return kOverflow;
  *passes = static_cast<std::uint8_t>(left >= right);
  return kOk;
}

int ciip_v1_quantize_q24_40(double value, std::int64_t* out) noexcept {
  if (out == nullptr) return kNull;
  if (!rounding_is_locked()) return kRoundingMode;
  if (!finite(value)) return kNonfinite;
  const double scaled_value = std::ldexp(value, 40);
  if (!finite(scaled_value)) return kOverflow;
  const double rounded = std::nearbyint(scaled_value);
  constexpr double kMin = -9223372036854775808.0;
  constexpr double kMaxExclusive = 9223372036854775808.0;
  if (!finite(rounded) || rounded < kMin || rounded >= kMaxExclusive) return kOverflow;
  *out = static_cast<std::int64_t>(rounded);
  return kOk;
}

int ciip_v1_jacobi_eigenvalues(const double upper_triangle[6], JacobiOutput* out) noexcept {
  if (upper_triangle == nullptr || out == nullptr) return kNull;
  if (!rounding_is_locked()) return kRoundingMode;
  for (int i = 0; i < 6; ++i) {
    if (!finite(upper_triangle[i])) return kNonfinite;
  }
  double matrix[3][3] = {
      {upper_triangle[0], upper_triangle[1], upper_triangle[2]},
      {upper_triangle[1], upper_triangle[3], upper_triangle[4]},
      {upper_triangle[2], upper_triangle[4], upper_triangle[5]},
  };
  constexpr int pivots[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    for (const auto& pivot : pivots) {
      if (!rotate(matrix, pivot[0], pivot[1])) return kNonfinite;
    }
  }

  const double trace = rn_add(rn_add(matrix[0][0], matrix[1][1]), matrix[2][2]);
  if (!finite(trace) || trace <= 0.0) return kNonfinite;
  const double max_offdiag = std::max({std::fabs(matrix[0][1]), std::fabs(matrix[0][2]),
                                       std::fabs(matrix[1][2])});
  const double tolerance = std::ldexp(trace, -40);
  if (!finite(max_offdiag) || !finite(tolerance) || max_offdiag > tolerance) return kResidual;

  struct EigenPair { double value; int axis; } eigen[3] = {
      {matrix[0][0], 0}, {matrix[1][1], 1}, {matrix[2][2], 2},
  };
  for (auto& item : eigen) {
    if (!finite(item.value)) return kNonfinite;
    if (item.value < 0.0) {
      if (std::fabs(item.value) <= tolerance) item.value = 0.0;
      else return kNegativeEigenvalue;
    }
  }
  std::sort(std::begin(eigen), std::end(eigen), [](const EigenPair& left, const EigenPair& right) {
    if (left.value != right.value) return left.value > right.value;
    return left.axis < right.axis;
  });
  out->lambda1 = eigen[0].value;
  out->lambda2 = eigen[1].value;
  out->lambda3 = eigen[2].value;
  out->trace = rn_add(rn_add(out->lambda1, out->lambda2), out->lambda3);
  out->max_offdiag = max_offdiag;
  return kOk;
}

int ciip_v1_anisotropy(const double eigenvalues[3], AnisotropyOutput* out) noexcept {
  if (eigenvalues == nullptr || out == nullptr) return kNull;
  if (!rounding_is_locked()) return kRoundingMode;
  const double lambda1 = eigenvalues[0], lambda2 = eigenvalues[1], lambda3 = eigenvalues[2];
  if (!finite(lambda1) || !finite(lambda2) || !finite(lambda3) ||
      lambda1 < lambda2 || lambda2 < lambda3 || lambda3 < 0.0) return kNonfinite;
  const double trace = rn_add(rn_add(lambda1, lambda2), lambda3);
  const double d12 = rn_sub(lambda1, lambda2);
  const double d23 = rn_sub(lambda2, lambda3);
  const double linearity = rn_div(d12, trace);
  const double planarity = rn_div(d23, trace);
  const double scalar = rn_mul(linearity, planarity);
  if (!finite(trace) || trace <= 0.0 || !finite(scalar) || scalar < 0.0 || scalar > 1.0) return kNonfinite;
  const double scaled_value = rn_mul(scalar, 65534.0);
  const double rounded = std::nearbyint(scaled_value);
  if (!finite(rounded) || rounded < 0.0 || rounded > 65534.0) return kOverflow;
  out->scalar = scalar;
  out->q = static_cast<std::uint16_t>(rounded);
  out->filtration = static_cast<std::uint16_t>(65534U - out->q);
  return kOk;
}

int ciip_v1_support_evidence(
    const std::uint8_t* occupied,
    const std::uint32_t* raw_counts,
    std::uint32_t grid_cell_count,
    const std::uint8_t* leaf_xyz,
    std::uint32_t leaf_count,
    std::uint8_t multiplier,
    std::uint32_t* support_count,
    std::int64_t* sum_x,
    std::int64_t* sum_y,
    std::int64_t* sum_z,
    std::int64_t* sum_xx,
    std::int64_t* sum_yy,
    std::int64_t* sum_zz,
    std::int64_t* sum_xy,
    std::int64_t* sum_xz,
    std::int64_t* sum_yz,
    std::uint32_t* raw_p10,
    std::uint32_t* raw_p90,
    std::uint32_t* balance_numerator,
    std::uint32_t* balance_denominator,
    std::uint32_t* component_count) noexcept {
  if (occupied == nullptr || raw_counts == nullptr || leaf_xyz == nullptr || support_count == nullptr ||
      sum_x == nullptr || sum_y == nullptr || sum_z == nullptr || sum_xx == nullptr || sum_yy == nullptr ||
      sum_zz == nullptr || sum_xy == nullptr || sum_xz == nullptr || sum_yz == nullptr || raw_p10 == nullptr ||
      raw_p90 == nullptr || balance_numerator == nullptr || balance_denominator == nullptr ||
      component_count == nullptr) return kNull;
  if (grid_cell_count != 32768 || leaf_count == 0 || leaf_count > 32768 ||
      (multiplier != 2 && multiplier != 3 && multiplier != 4)) return kOverflow;

  std::array<std::uint8_t, 32768> seen{};
  std::uint32_t occupied_total = 0;
  for (std::uint32_t index = 0; index < 32768; ++index) {
    if (occupied[index] > 1) return kOverflow;
    if (occupied[index] == 0) {
      if (raw_counts[index] != 0) return kOverflow;
    } else {
      if (raw_counts[index] == 0) return kOverflow;
      ++occupied_total;
    }
  }
  if (occupied_total != leaf_count) return kOverflow;
  std::uint32_t previous_morton = 0;
  for (std::uint32_t leaf = 0; leaf < leaf_count; ++leaf) {
    const std::uint8_t x = leaf_xyz[3 * leaf];
    const std::uint8_t y = leaf_xyz[3 * leaf + 1];
    const std::uint8_t z = leaf_xyz[3 * leaf + 2];
    if (x >= 32 || y >= 32 || z >= 32) return kOverflow;
    const std::uint32_t morton = morton5(x, y, z);
    if (occupied[morton] == 0 || seen[morton] != 0 || (leaf > 0 && morton <= previous_morton)) return kOverflow;
    seen[morton] = 1;
    previous_morton = morton;
  }

  constexpr int kMaxStencil = 729;
  std::array<std::uint32_t, kMaxStencil> counts{};
  std::array<std::uint8_t, kMaxStencil> local_mask{};
  std::array<std::uint8_t, kMaxStencil> visited{};
  std::array<std::uint16_t, kMaxStencil> queue{};
  const int radius = static_cast<int>(multiplier);
  const int width = 2 * radius + 1;
  const int radius_squared = radius * radius;

  for (std::uint32_t leaf = 0; leaf < leaf_count; ++leaf) {
    const int center_x = leaf_xyz[3 * leaf];
    const int center_y = leaf_xyz[3 * leaf + 1];
    const int center_z = leaf_xyz[3 * leaf + 2];
    std::fill(local_mask.begin(), local_mask.end(), 0);
    std::fill(visited.begin(), visited.end(), 0);
    std::uint32_t count = 0;
    std::uint32_t plus = 0;
    std::uint32_t minus = 0;
    i128 moments[9] = {};

    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          if (dx * dx + dy * dy + dz * dz > radius_squared) continue;
          const int x = center_x + dx, y = center_y + dy, z = center_z + dz;
          if (x < 0 || x >= 32 || y < 0 || y >= 32 || z < 0 || z >= 32) continue;
          const std::uint32_t morton = morton5(static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(y),
                                               static_cast<std::uint8_t>(z));
          if (occupied[morton] == 0) continue;
          if (count >= kMaxStencil) return kOverflow;
          counts[count++] = raw_counts[morton];
          const int local_x = dx + radius, local_y = dy + radius, local_z = dz + radius;
          local_mask[local_x + width * (local_y + width * local_z)] = 1;
          const i128 ix = static_cast<i128>(x), iy = static_cast<i128>(y), iz = static_cast<i128>(z);
          if (!accumulate_i128(&moments[0], ix) || !accumulate_i128(&moments[1], iy) ||
              !accumulate_i128(&moments[2], iz) || !accumulate_i128(&moments[3], ix * ix) ||
              !accumulate_i128(&moments[4], iy * iy) || !accumulate_i128(&moments[5], iz * iz) ||
              !accumulate_i128(&moments[6], ix * iy) || !accumulate_i128(&moments[7], ix * iz) ||
              !accumulate_i128(&moments[8], iy * iz)) return kOverflow;
          if (dx != 0 || dy != 0 || dz != 0) {
            const int first = dx != 0 ? dx : (dy != 0 ? dy : dz);
            if (first > 0) ++plus;
            else ++minus;
          }
        }
      }
    }
    if (count == 0) return kOverflow;
    support_count[leaf] = count;
    std::int64_t* outputs[9] = {
        &sum_x[leaf], &sum_y[leaf], &sum_z[leaf], &sum_xx[leaf], &sum_yy[leaf],
        &sum_zz[leaf], &sum_xy[leaf], &sum_xz[leaf], &sum_yz[leaf],
    };
    for (int index = 0; index < 9; ++index) {
      if (!emit_i64(moments[index], outputs[index])) return kOverflow;
    }
    std::sort(counts.begin(), counts.begin() + count);
    const std::uint32_t p10_index = (count + 9) / 10 - 1;
    const std::uint32_t p90_index = (9 * count + 9) / 10 - 1;
    raw_p10[leaf] = counts[p10_index];
    raw_p90[leaf] = counts[p90_index];
    balance_numerator[leaf] = std::min(plus, minus);
    balance_denominator[leaf] = std::max(plus, minus);

    std::uint32_t components = 0;
    const int local_cells = width * width * width;
    for (int seed = 0; seed < local_cells; ++seed) {
      if (local_mask[seed] == 0 || visited[seed] != 0) continue;
      ++components;
      std::uint16_t head = 0, tail = 0;
      queue[tail++] = static_cast<std::uint16_t>(seed);
      visited[seed] = 1;
      while (head < tail) {
        const int current = queue[head++];
        const int lx = current % width;
        const int ly = (current / width) % width;
        const int lz = current / (width * width);
        constexpr int directions[6][3] = {
            {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        };
        for (const auto& direction : directions) {
          const int nx = lx + direction[0], ny = ly + direction[1], nz = lz + direction[2];
          if (nx < 0 || nx >= width || ny < 0 || ny >= width || nz < 0 || nz >= width) continue;
          const int neighbor = nx + width * (ny + width * nz);
          if (local_mask[neighbor] != 0 && visited[neighbor] == 0) {
            if (tail >= kMaxStencil) return kOverflow;
            visited[neighbor] = 1;
            queue[tail++] = static_cast<std::uint16_t>(neighbor);
          }
        }
      }
    }
    component_count[leaf] = components;
  }
  return kOk;
}

// Evaluate exactly one support-dependent gate stage for a survivor mask.  The
// caller advances the mask only after consuming `passes`, so a rejected row is
// never evaluated by a later stage.  Stage numbers are the frozen stage bits;
// growth (3) and numeric tensor construction (6) have dedicated batch APIs.
int ciip_v1_support_stage(
    const std::uint8_t* occupied,
    const std::uint32_t* raw_counts,
    std::uint32_t grid_cell_count,
    const std::uint8_t* leaf_xyz,
    std::uint32_t leaf_count,
    std::uint8_t multiplier,
    std::uint8_t stage,
    const std::uint8_t* active,
    std::uint32_t* support_count,
    std::int64_t* sum_x,
    std::int64_t* sum_y,
    std::int64_t* sum_z,
    std::int64_t* sum_xx,
    std::int64_t* sum_yy,
    std::int64_t* sum_zz,
    std::int64_t* sum_xy,
    std::int64_t* sum_xz,
    std::int64_t* sum_yz,
    std::uint32_t* raw_p10,
    std::uint32_t* raw_p90,
    std::uint32_t* balance_numerator,
    std::uint32_t* balance_denominator,
    std::uint32_t* component_count,
    std::uint8_t* passes) noexcept {
  if (occupied == nullptr || raw_counts == nullptr || leaf_xyz == nullptr || active == nullptr ||
      support_count == nullptr || sum_x == nullptr || sum_y == nullptr || sum_z == nullptr ||
      sum_xx == nullptr || sum_yy == nullptr || sum_zz == nullptr || sum_xy == nullptr ||
      sum_xz == nullptr || sum_yz == nullptr || raw_p10 == nullptr || raw_p90 == nullptr ||
      balance_numerator == nullptr || balance_denominator == nullptr || component_count == nullptr ||
      passes == nullptr) return kNull;
  if (grid_cell_count != 32768 || leaf_count == 0 || leaf_count > 32768 ||
      (multiplier != 2 && multiplier != 3 && multiplier != 4) ||
      (stage != 0 && stage != 1 && stage != 2 && stage != 4 && stage != 5)) return kOverflow;

  std::array<std::uint8_t, 32768> seen{};
  std::uint32_t occupied_total = 0;
  for (std::uint32_t index = 0; index < 32768; ++index) {
    if (occupied[index] > 1) return kOverflow;
    if (occupied[index] == 0) {
      if (raw_counts[index] != 0) return kOverflow;
    } else {
      if (raw_counts[index] == 0) return kOverflow;
      ++occupied_total;
    }
  }
  if (occupied_total != leaf_count) return kOverflow;
  std::uint32_t previous_morton = 0;
  for (std::uint32_t leaf = 0; leaf < leaf_count; ++leaf) {
    if (active[leaf] > 1) return kOverflow;
    const std::uint8_t x = leaf_xyz[3 * leaf];
    const std::uint8_t y = leaf_xyz[3 * leaf + 1];
    const std::uint8_t z = leaf_xyz[3 * leaf + 2];
    if (x >= 32 || y >= 32 || z >= 32) return kOverflow;
    const std::uint32_t morton = morton5(x, y, z);
    if (occupied[morton] == 0 || seen[morton] != 0 || (leaf > 0 && morton <= previous_morton)) {
      return kOverflow;
    }
    seen[morton] = 1;
    previous_morton = morton;
  }

  constexpr int kMaxStencil = 729;
  std::array<std::uint32_t, kMaxStencil> counts{};
  std::array<std::uint8_t, kMaxStencil> local_mask{};
  std::array<std::uint8_t, kMaxStencil> visited{};
  std::array<std::uint16_t, kMaxStencil> queue{};
  const int radius = static_cast<int>(multiplier);
  const int width = 2 * radius + 1;
  const int radius_squared = radius * radius;
  const std::uint32_t minimum_support = multiplier == 2 ? 9U : (multiplier == 3 ? 21U : 37U);

  for (std::uint32_t leaf = 0; leaf < leaf_count; ++leaf) {
    passes[leaf] = 0;
    if (active[leaf] == 0) continue;
    const int center_x = leaf_xyz[3 * leaf];
    const int center_y = leaf_xyz[3 * leaf + 1];
    const int center_z = leaf_xyz[3 * leaf + 2];
    if (stage == 1) {
      std::fill(local_mask.begin(), local_mask.end(), 0);
      std::fill(visited.begin(), visited.end(), 0);
    }
    std::uint32_t count = 0;
    std::uint32_t plus = 0;
    std::uint32_t minus = 0;
    i128 moments[9] = {};
    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          if (dx * dx + dy * dy + dz * dz > radius_squared) continue;
          const int x = center_x + dx, y = center_y + dy, z = center_z + dz;
          if (x < 0 || x >= 32 || y < 0 || y >= 32 || z < 0 || z >= 32) continue;
          const std::uint32_t morton = morton5(static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(y),
                                               static_cast<std::uint8_t>(z));
          if (occupied[morton] == 0) continue;
          if (count >= kMaxStencil) return kOverflow;
          if (stage == 4) counts[count] = raw_counts[morton];
          if (stage == 1) {
            const int local_x = dx + radius, local_y = dy + radius, local_z = dz + radius;
            local_mask[local_x + width * (local_y + width * local_z)] = 1;
          } else if (stage == 2 && (dx != 0 || dy != 0 || dz != 0)) {
            const int first = dx != 0 ? dx : (dy != 0 ? dy : dz);
            if (first > 0) ++plus;
            else ++minus;
          } else if (stage == 5) {
            const i128 ix = static_cast<i128>(x), iy = static_cast<i128>(y), iz = static_cast<i128>(z);
            if (!accumulate_i128(&moments[0], ix) || !accumulate_i128(&moments[1], iy) ||
                !accumulate_i128(&moments[2], iz) || !accumulate_i128(&moments[3], ix * ix) ||
                !accumulate_i128(&moments[4], iy * iy) || !accumulate_i128(&moments[5], iz * iz) ||
                !accumulate_i128(&moments[6], ix * iy) || !accumulate_i128(&moments[7], ix * iz) ||
                !accumulate_i128(&moments[8], iy * iz)) return kOverflow;
          }
          ++count;
        }
      }
    }
    if (count == 0) return kOverflow;

    if (stage == 0) {
      support_count[leaf] = count;
      passes[leaf] = static_cast<std::uint8_t>(count >= minimum_support);
    } else if (stage == 1) {
      std::uint32_t components = 0;
      const int local_cells = width * width * width;
      for (int seed = 0; seed < local_cells; ++seed) {
        if (local_mask[seed] == 0 || visited[seed] != 0) continue;
        ++components;
        std::uint16_t head = 0, tail = 0;
        queue[tail++] = static_cast<std::uint16_t>(seed);
        visited[seed] = 1;
        while (head < tail) {
          const int current = queue[head++];
          const int lx = current % width;
          const int ly = (current / width) % width;
          const int lz = current / (width * width);
          constexpr int directions[6][3] = {
              {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
          };
          for (const auto& direction : directions) {
            const int nx = lx + direction[0], ny = ly + direction[1], nz = lz + direction[2];
            if (nx < 0 || nx >= width || ny < 0 || ny >= width || nz < 0 || nz >= width) continue;
            const int neighbor = nx + width * (ny + width * nz);
            if (local_mask[neighbor] != 0 && visited[neighbor] == 0) {
              if (tail >= kMaxStencil) return kOverflow;
              visited[neighbor] = 1;
              queue[tail++] = static_cast<std::uint16_t>(neighbor);
            }
          }
        }
      }
      component_count[leaf] = components;
      passes[leaf] = static_cast<std::uint8_t>(components == 1);
    } else if (stage == 2) {
      const std::uint32_t numerator = std::min(plus, minus);
      const std::uint32_t denominator = std::max(plus, minus);
      balance_numerator[leaf] = numerator;
      balance_denominator[leaf] = denominator;
      u128 left = 0, right = 0;
      if (denominator == 0 || numerator > denominator ||
          __builtin_mul_overflow(static_cast<u128>(20), static_cast<u128>(numerator), &left) ||
          __builtin_mul_overflow(static_cast<u128>(11), static_cast<u128>(denominator), &right)) return kOverflow;
      passes[leaf] = static_cast<std::uint8_t>(left >= right);
    } else if (stage == 4) {
      std::sort(counts.begin(), counts.begin() + count);
      const std::uint32_t p10_index = (count + 9) / 10 - 1;
      const std::uint32_t p90_index = (9 * count + 9) / 10 - 1;
      raw_p10[leaf] = counts[p10_index];
      raw_p90[leaf] = counts[p90_index];
      const std::uint64_t denominator = std::max<std::uint64_t>(1, raw_p10[leaf]);
      u128 limit = 0;
      if (__builtin_mul_overflow(static_cast<u128>(32), static_cast<u128>(denominator), &limit)) return kOverflow;
      passes[leaf] = static_cast<std::uint8_t>(static_cast<u128>(raw_p90[leaf]) <= limit);
    } else {
      std::int64_t* outputs[9] = {
          &sum_x[leaf], &sum_y[leaf], &sum_z[leaf], &sum_xx[leaf], &sum_yy[leaf],
          &sum_zz[leaf], &sum_xy[leaf], &sum_xz[leaf], &sum_yz[leaf],
      };
      for (int index = 0; index < 9; ++index) {
        if (!emit_i64(moments[index], outputs[index])) return kOverflow;
      }
      passes[leaf] = 1;
    }
  }
  return kOk;
}

int ciip_v1_growth_gate_masked(
    const std::uint32_t* n1,
    const std::uint32_t* n2,
    const std::uint32_t* n3,
    const std::uint8_t* active,
    std::uint32_t count,
    std::int32_t* growth12,
    std::int32_t* growth23,
    std::uint8_t* passes) noexcept {
  if (n1 == nullptr || n2 == nullptr || n3 == nullptr || active == nullptr || growth12 == nullptr ||
      growth23 == nullptr || passes == nullptr || count == 0 || count > 32768) return kNull;
  for (std::uint32_t index = 0; index < count; ++index) {
    passes[index] = 0;
    if (active[index] > 1) return kOverflow;
    if (active[index] == 0) continue;
    const int status = ciip_v1_growth_gate(n1[index], n2[index], n3[index],
                                           &growth12[index], &growth23[index]);
    if (status != kOk) return status;
    passes[index] = static_cast<std::uint8_t>(growth12[index] == 0 && growth23[index] == 0);
  }
  return kOk;
}

int ciip_v1_moments_to_field_masked(
    const std::uint32_t* support_count,
    const std::int64_t* sum_x,
    const std::int64_t* sum_y,
    const std::int64_t* sum_z,
    const std::int64_t* sum_xx,
    const std::int64_t* sum_yy,
    const std::int64_t* sum_zz,
    const std::int64_t* sum_xy,
    const std::int64_t* sum_xz,
    const std::int64_t* sum_yz,
    const std::uint8_t* active,
    std::uint32_t count,
    std::int64_t* trace_q,
    std::int64_t* lambda1_q,
    std::int64_t* lambda2_q,
    std::int64_t* lambda3_q,
    std::uint16_t* anisotropy_q,
    std::uint16_t* filtration,
    std::uint8_t* result_code) noexcept {
  if (support_count == nullptr || sum_x == nullptr || sum_y == nullptr || sum_z == nullptr ||
      sum_xx == nullptr || sum_yy == nullptr || sum_zz == nullptr || sum_xy == nullptr ||
      sum_xz == nullptr || sum_yz == nullptr || active == nullptr || trace_q == nullptr ||
      lambda1_q == nullptr || lambda2_q == nullptr || lambda3_q == nullptr ||
      anisotropy_q == nullptr || filtration == nullptr || result_code == nullptr ||
      count == 0 || count > 98304) return kNull;
  if (!rounding_is_locked()) return kRoundingMode;

  for (std::uint32_t index = 0; index < count; ++index) {
    result_code[index] = 0;
    if (active[index] > 1) return kOverflow;
    if (active[index] == 0) continue;
    const std::uint32_t n = support_count[index];
    if (n == 0) {
      result_code[index] = 7;
      continue;
    }
    const double denominator = static_cast<double>(n);
    const double means[3] = {
        rn_div(static_cast<double>(sum_x[index]), denominator),
        rn_div(static_cast<double>(sum_y[index]), denominator),
        rn_div(static_cast<double>(sum_z[index]), denominator),
    };
    const double cube = rn_div(1.0, 12.0);
    const double upper[6] = {
        rn_add(rn_sub(rn_div(static_cast<double>(sum_xx[index]), denominator),
                      rn_mul(means[0], means[0])), cube),
        rn_sub(rn_div(static_cast<double>(sum_xy[index]), denominator), rn_mul(means[0], means[1])),
        rn_sub(rn_div(static_cast<double>(sum_xz[index]), denominator), rn_mul(means[0], means[2])),
        rn_add(rn_sub(rn_div(static_cast<double>(sum_yy[index]), denominator),
                      rn_mul(means[1], means[1])), cube),
        rn_sub(rn_div(static_cast<double>(sum_yz[index]), denominator), rn_mul(means[1], means[2])),
        rn_add(rn_sub(rn_div(static_cast<double>(sum_zz[index]), denominator),
                      rn_mul(means[2], means[2])), cube),
    };
    const double trace = rn_add(rn_add(upper[0], upper[3]), upper[5]);
    if (!finite(trace) || trace <= 0.0) {
      result_code[index] = 6;
      continue;
    }
    int status = ciip_v1_quantize_q24_40(trace, &trace_q[index]);
    if (status != kOk) {
      result_code[index] = 7;
      continue;
    }
    JacobiOutput jacobi{};
    status = ciip_v1_jacobi_eigenvalues(upper, &jacobi);
    if (status != kOk) {
      result_code[index] = 7;
      continue;
    }
    AnisotropyOutput anisotropy{};
    const double eigenvalues[3] = {jacobi.lambda1, jacobi.lambda2, jacobi.lambda3};
    status = ciip_v1_anisotropy(eigenvalues, &anisotropy);
    if (status != kOk || ciip_v1_quantize_q24_40(eigenvalues[0], &lambda1_q[index]) != kOk ||
        ciip_v1_quantize_q24_40(eigenvalues[1], &lambda2_q[index]) != kOk ||
        ciip_v1_quantize_q24_40(eigenvalues[2], &lambda3_q[index]) != kOk) {
      result_code[index] = 7;
      continue;
    }
    std::int64_t partial = 0, exact_trace = 0;
    if (__builtin_add_overflow(lambda1_q[index], lambda2_q[index], &partial) ||
        __builtin_add_overflow(partial, lambda3_q[index], &exact_trace)) {
      result_code[index] = 7;
      continue;
    }
    trace_q[index] = exact_trace;
    anisotropy_q[index] = anisotropy.q;
    filtration[index] = anisotropy.filtration;
  }
  return kOk;
}

int ciip_v1_build_adjacency(
    const std::uint8_t case_digest[32],
    const std::uint64_t* mortons,
    std::uint32_t occupied_count,
    EdgeRow* output_rows,
    std::uint32_t output_capacity,
    std::uint32_t* output_count) noexcept {
  if (case_digest == nullptr || mortons == nullptr || output_rows == nullptr || output_count == nullptr) {
    return kNull;
  }
  std::vector<EdgeRow> rows;
  const int status = build_adjacency_rows(case_digest, mortons, occupied_count, &rows);
  if (status != kOk) return status;
  if (rows.size() > output_capacity) return kOverflow;
  if (!rows.empty()) std::memcpy(output_rows, rows.data(), rows.size() * sizeof(EdgeRow));
  *output_count = static_cast<std::uint32_t>(rows.size());
  return kOk;
}

int ciip_v1_h0_persistence(
    const std::uint8_t case_digest[32],
    std::uint8_t arm,
    const FieldRow* fields,
    std::uint32_t vertex_count,
    const EdgeRow* edges,
    std::uint32_t edge_count,
    PersistenceRow* output,
    std::uint32_t* attributed_count) noexcept {
  if (case_digest == nullptr || fields == nullptr || edges == nullptr || output == nullptr ||
      attributed_count == nullptr || arm > 1 || vertex_count == 0 || vertex_count % 3 != 0 ||
      vertex_count > 98304 || edge_count == 0 || edge_count > 360448) return kNull;
  const std::uint32_t occupied_count = vertex_count / 3;
  std::vector<std::uint64_t> mortons(occupied_count);
  struct FieldCollision {
    std::array<std::uint8_t, 16> digest;
    std::uint8_t scale;
    std::uint64_t morton;
  };
  std::vector<FieldCollision> field_collisions(vertex_count);
  for (std::uint32_t index = 0; index < vertex_count; ++index) {
    const std::uint8_t expected_scale = static_cast<std::uint8_t>(index / occupied_count);
    if (fields[index].arm != arm || fields[index].scale != expected_scale || fields[index].status > 1 ||
        fields[index].filtration > 65534 || fields[index].flags != 0) return kOverflow;
    if (index < occupied_count) {
      if (fields[index].morton >= (1U << 15) ||
          (index != 0 && fields[index].morton <= fields[index - 1].morton)) return kOverflow;
      mortons[index] = fields[index].morton;
    } else if (fields[index].morton != mortons[index % occupied_count]) {
      return kOverflow;
    }
    std::uint8_t expected_id[16];
    hash_field(case_digest, arm, expected_scale, fields[index].morton, expected_id);
    if (std::memcmp(expected_id, fields[index].row_id, 16) != 0) return kOverflow;
    std::memcpy(field_collisions[index].digest.data(), fields[index].row_id, 16);
    field_collisions[index].scale = expected_scale;
    field_collisions[index].morton = fields[index].morton;
  }
  std::sort(field_collisions.begin(), field_collisions.end(),
            [](const FieldCollision& left, const FieldCollision& right) {
    const int digest_order = std::memcmp(left.digest.data(), right.digest.data(), 16);
    if (digest_order != 0) return digest_order < 0;
    if (left.scale != right.scale) return left.scale < right.scale;
    return left.morton < right.morton;
  });
  for (std::size_t index = 1; index < field_collisions.size(); ++index) {
    if (field_collisions[index - 1].digest == field_collisions[index].digest &&
        (field_collisions[index - 1].scale != field_collisions[index].scale ||
         field_collisions[index - 1].morton != field_collisions[index].morton)) return kDigestCollision;
  }

  std::vector<EdgeRow> expected_edges;
  int status = build_adjacency_rows(case_digest, mortons.data(), occupied_count, &expected_edges);
  if (status != kOk) return status;
  if (expected_edges.size() != edge_count ||
      std::memcmp(expected_edges.data(), edges, static_cast<std::size_t>(edge_count) * sizeof(EdgeRow)) != 0) {
    return kOverflow;
  }

  std::memset(output, 0, static_cast<std::size_t>(vertex_count) * sizeof(PersistenceRow));
  std::vector<std::uint16_t> edge_f(edge_count);
  std::vector<std::uint32_t> vertex_counts(65536, 0), edge_counts(65536, 0);
  for (std::uint32_t index = 0; index < vertex_count; ++index) ++vertex_counts[fields[index].filtration];
  for (std::uint32_t index = 0; index < edge_count; ++index) {
    edge_f[index] = std::max(fields[edges[index].lower].filtration, fields[edges[index].upper].filtration);
    ++edge_counts[edge_f[index]];
  }
  std::vector<std::uint32_t> vertex_offsets(65537, 0), edge_offsets(65537, 0);
  for (std::size_t f = 0; f < 65536; ++f) {
    vertex_offsets[f + 1] = vertex_offsets[f] + vertex_counts[f];
    edge_offsets[f + 1] = edge_offsets[f] + edge_counts[f];
  }
  std::vector<std::uint32_t> vertex_next(vertex_offsets.begin(), vertex_offsets.end() - 1);
  std::vector<std::uint32_t> edge_next(edge_offsets.begin(), edge_offsets.end() - 1);
  std::vector<std::uint32_t> vertex_order(vertex_count), edge_order(edge_count);
  for (std::uint32_t index = 0; index < vertex_count; ++index) {
    vertex_order[vertex_next[fields[index].filtration]++] = index;
  }
  for (std::uint32_t index = 0; index < edge_count; ++index) {
    edge_order[edge_next[edge_f[index]]++] = index;
  }

  std::vector<std::uint32_t> parent(vertex_count), owner(vertex_count), spatial_count(vertex_count, 0);
  std::vector<std::uint8_t> rank(vertex_count, 0), active(vertex_count, 0), scale_mask(vertex_count, 0),
      all_valid(vertex_count, 1), emitted(vertex_count, 0);
  for (std::uint32_t index = 0; index < vertex_count; ++index) {
    parent[index] = index;
    owner[index] = index;
  }
  auto find = [&parent](std::uint32_t value) {
    std::uint32_t root = value;
    while (parent[root] != root) root = parent[root];
    while (parent[value] != value) {
      const std::uint32_t next = parent[value];
      parent[value] = root;
      value = next;
    }
    return root;
  };

  struct EssentialCollision {
    std::array<std::uint8_t, 16> digest;
    std::array<std::uint8_t, 16> birth;
  };
  struct IntervalCollision {
    std::array<std::uint8_t, 16> digest;
    std::array<std::uint8_t, 16> birth;
    std::array<std::uint8_t, 16> death;
    std::uint16_t birth_f;
    std::uint16_t death_f;
  };
  std::vector<EssentialCollision> essential_collisions;
  std::vector<IntervalCollision> interval_collisions;
  essential_collisions.reserve(vertex_count);
  interval_collisions.reserve(vertex_count);
  std::uint32_t attributed = 0;
  auto emit = [&](std::uint32_t birth_vertex, std::uint16_t death_f, std::uint8_t disposition,
                  const std::uint8_t death_id[16], std::uint32_t root) -> int {
    if (emitted[birth_vertex] != 0) return kOverflow;
    const std::uint16_t birth_f = fields[birth_vertex].filtration;
    const std::uint16_t lifetime = static_cast<std::uint16_t>(death_f - birth_f);
    std::uint16_t reason = 0;
    if (fields[birth_vertex].anisotropy_q == 0) reason = 1;
    else if (lifetime < 2) reason = 2;
    else if (__builtin_popcount(static_cast<unsigned>(scale_mask[root])) < 2) reason = 3;
    else if (spatial_count[root] < 1) reason = 4;
    else if (all_valid[root] == 0) reason = 5;
    PersistenceRow& row = output[birth_vertex];
    std::memcpy(row.birth_id, fields[birth_vertex].row_id, 16);
    std::memcpy(row.death_id, death_id, 16);
    hash_interval(case_digest, arm, row.birth_id, row.death_id, birth_f, death_f, row.interval_id);
    row.arm = arm;
    row.status = disposition;
    row.reason = reason;
    row.birth_f = birth_f;
    row.death_f = death_f;
    row.lifetime = lifetime;
    row.scale_mask = scale_mask[root];
    row.spatial_count = spatial_count[root];
    emitted[birth_vertex] = 1;
    IntervalCollision collision{};
    std::memcpy(collision.digest.data(), row.interval_id, 16);
    std::memcpy(collision.birth.data(), row.birth_id, 16);
    std::memcpy(collision.death.data(), row.death_id, 16);
    collision.birth_f = birth_f;
    collision.death_f = death_f;
    interval_collisions.push_back(collision);
    if (reason == 0) ++attributed;
    return kOk;
  };

  for (std::uint32_t f = 0; f < 65535; ++f) {
    for (std::uint32_t position = vertex_offsets[f]; position < vertex_offsets[f + 1]; ++position) {
      const std::uint32_t vertex = vertex_order[position];
      active[vertex] = 1;
      scale_mask[vertex] = static_cast<std::uint8_t>(1U << fields[vertex].scale);
      all_valid[vertex] = static_cast<std::uint8_t>(fields[vertex].status == 0);
    }
    for (std::uint32_t position = edge_offsets[f]; position < edge_offsets[f + 1]; ++position) {
      const std::uint32_t edge_index = edge_order[position];
      const EdgeRow& edge = edges[edge_index];
      if (active[edge.lower] == 0 || active[edge.upper] == 0) return kOverflow;
      std::uint32_t root_left = find(edge.lower), root_right = find(edge.upper);
      const bool spatial = edge.kind == 0;
      if (root_left == root_right) {
        if (spatial) ++spatial_count[root_left];
        continue;
      }
      const std::uint32_t owner_left = owner[root_left], owner_right = owner[root_right];
      const bool left_elder = fields[owner_left].filtration < fields[owner_right].filtration ||
          (fields[owner_left].filtration == fields[owner_right].filtration && owner_left < owner_right);
      const std::uint32_t survivor_owner = left_elder ? owner_left : owner_right;
      const std::uint32_t loser_owner = left_elder ? owner_right : owner_left;
      const std::uint32_t loser_root = left_elder ? root_right : root_left;
      status = emit(loser_owner, static_cast<std::uint16_t>(f), 0, edge.edge_id, loser_root);
      if (status != kOk) return status;
      if (rank[root_left] < rank[root_right] ||
          (rank[root_left] == rank[root_right] && root_right < root_left)) std::swap(root_left, root_right);
      parent[root_right] = root_left;
      if (rank[root_left] == rank[root_right]) ++rank[root_left];
      owner[root_left] = survivor_owner;
      scale_mask[root_left] = static_cast<std::uint8_t>(scale_mask[root_left] | scale_mask[root_right]);
      spatial_count[root_left] = spatial_count[root_left] + spatial_count[root_right] + (spatial ? 1U : 0U);
      all_valid[root_left] = static_cast<std::uint8_t>(all_valid[root_left] != 0 && all_valid[root_right] != 0);
    }
  }

  for (std::uint32_t root = 0; root < vertex_count; ++root) {
    if (find(root) != root) continue;
    const std::uint32_t birth_vertex = owner[root];
    std::uint8_t death_id[16];
    hash_essential(case_digest, arm, fields[birth_vertex].row_id, death_id);
    EssentialCollision collision{};
    std::memcpy(collision.digest.data(), death_id, 16);
    std::memcpy(collision.birth.data(), fields[birth_vertex].row_id, 16);
    essential_collisions.push_back(collision);
    status = emit(birth_vertex, 65535, 1, death_id, root);
    if (status != kOk) return status;
  }
  if (std::find(emitted.begin(), emitted.end(), static_cast<std::uint8_t>(0)) != emitted.end()) return kOverflow;

  auto digest_less = [](const auto& left, const auto& right) {
    return std::memcmp(left.digest.data(), right.digest.data(), 16) < 0;
  };
  std::sort(essential_collisions.begin(), essential_collisions.end(), digest_less);
  for (std::size_t index = 1; index < essential_collisions.size(); ++index) {
    if (essential_collisions[index - 1].digest == essential_collisions[index].digest &&
        essential_collisions[index - 1].birth != essential_collisions[index].birth) return kDigestCollision;
  }
  std::sort(interval_collisions.begin(), interval_collisions.end(), digest_less);
  for (std::size_t index = 1; index < interval_collisions.size(); ++index) {
    const auto& left = interval_collisions[index - 1];
    const auto& right = interval_collisions[index];
    if (left.digest == right.digest &&
        (left.birth != right.birth || left.death != right.death || left.birth_f != right.birth_f ||
         left.death_f != right.death_f)) return kDigestCollision;
  }
  *attributed_count = attributed;
  return kOk;
}

int ciip_v1_validate_persistence_collision_domain(
    const std::uint8_t case_digest[32],
    const FieldRow* fields,
    const PersistenceRow* persistence,
    std::uint32_t row_count) noexcept {
  if (case_digest == nullptr || fields == nullptr || persistence == nullptr || row_count == 0 ||
      row_count > 196608 || row_count % 6 != 0) return kNull;
  struct FieldEntry {
    std::array<std::uint8_t, 16> digest;
    std::uint8_t arm;
    std::uint8_t scale;
    std::uint64_t morton;
  };
  struct EssentialEntry {
    std::array<std::uint8_t, 16> digest;
    std::uint8_t arm;
    std::array<std::uint8_t, 16> birth;
  };
  struct IntervalEntry {
    std::array<std::uint8_t, 16> digest;
    std::uint8_t arm;
    std::array<std::uint8_t, 16> birth;
    std::array<std::uint8_t, 16> death;
    std::uint16_t birth_f;
    std::uint16_t death_f;
  };
  std::vector<FieldEntry> field_entries(row_count);
  std::vector<EssentialEntry> essential_entries;
  std::vector<IntervalEntry> interval_entries(row_count);
  essential_entries.reserve(row_count);
  for (std::uint32_t index = 0; index < row_count; ++index) {
    const FieldRow& field = fields[index];
    const PersistenceRow& row = persistence[index];
    if (field.arm > 1 || field.scale > 2 || row.arm != field.arm ||
        std::memcmp(row.birth_id, field.row_id, 16) != 0) return kOverflow;
    std::uint8_t expected_field[16];
    hash_field(case_digest, field.arm, field.scale, field.morton, expected_field);
    if (std::memcmp(expected_field, field.row_id, 16) != 0) return kOverflow;
    if (row.status == 1) {
      std::uint8_t expected_death[16];
      hash_essential(case_digest, row.arm, row.birth_id, expected_death);
      if (std::memcmp(expected_death, row.death_id, 16) != 0 || row.death_f != 65535) return kOverflow;
      EssentialEntry essential{};
      std::memcpy(essential.digest.data(), row.death_id, 16);
      essential.arm = row.arm;
      std::memcpy(essential.birth.data(), row.birth_id, 16);
      essential_entries.push_back(essential);
    } else if (row.status != 0) {
      return kOverflow;
    }
    std::uint8_t expected_interval[16];
    hash_interval(case_digest, row.arm, row.birth_id, row.death_id,
                  row.birth_f, row.death_f, expected_interval);
    if (std::memcmp(expected_interval, row.interval_id, 16) != 0) return kOverflow;
    std::memcpy(field_entries[index].digest.data(), field.row_id, 16);
    field_entries[index].arm = field.arm;
    field_entries[index].scale = field.scale;
    field_entries[index].morton = field.morton;
    std::memcpy(interval_entries[index].digest.data(), row.interval_id, 16);
    interval_entries[index].arm = row.arm;
    std::memcpy(interval_entries[index].birth.data(), row.birth_id, 16);
    std::memcpy(interval_entries[index].death.data(), row.death_id, 16);
    interval_entries[index].birth_f = row.birth_f;
    interval_entries[index].death_f = row.death_f;
  }
  auto digest_less = [](const auto& left, const auto& right) {
    return std::memcmp(left.digest.data(), right.digest.data(), 16) < 0;
  };
  std::sort(field_entries.begin(), field_entries.end(), digest_less);
  for (std::size_t index = 1; index < field_entries.size(); ++index) {
    const auto& left = field_entries[index - 1];
    const auto& right = field_entries[index];
    if (left.digest == right.digest &&
        (left.arm != right.arm || left.scale != right.scale || left.morton != right.morton)) {
      return kDigestCollision;
    }
  }
  std::sort(essential_entries.begin(), essential_entries.end(), digest_less);
  for (std::size_t index = 1; index < essential_entries.size(); ++index) {
    const auto& left = essential_entries[index - 1];
    const auto& right = essential_entries[index];
    if (left.digest == right.digest && (left.arm != right.arm || left.birth != right.birth)) {
      return kDigestCollision;
    }
  }
  std::sort(interval_entries.begin(), interval_entries.end(), digest_less);
  for (std::size_t index = 1; index < interval_entries.size(); ++index) {
    const auto& left = interval_entries[index - 1];
    const auto& right = interval_entries[index];
    if (left.digest == right.digest &&
        (left.arm != right.arm || left.birth != right.birth || left.death != right.death ||
         left.birth_f != right.birth_f || left.death_f != right.death_f)) return kDigestCollision;
  }
  return kOk;
}

int ciip_v1_validate_known_answers() noexcept {
  const std::uint8_t zero_source[32] = {};
  Sha256 case_sha = tagged_sha("CIIP_CASE_V1");
  case_sha.update(zero_source, sizeof(zero_source));
  const auto case_digest = case_sha.finish();
  if (!digest_matches_hex(case_digest.data(), 32,
      "6392da896a8301fd064f0c7193e192ecefb52d4849dfd249b2c561817aabce47")) return kOverflow;

  std::uint8_t baseline_field[16], isotropic_field[16], edge[16], death[16], interval[16];
  hash_field(case_digest.data(), 0, 1, 53, baseline_field);
  if (!digest_matches_hex(baseline_field, 16, "ae35aeb5777676ad6ac61aba20f90caa")) return kOverflow;
  hash_field(case_digest.data(), 1, 1, 53, isotropic_field);
  if (!digest_matches_hex(isotropic_field, 16, "72dc258326cb1f01014240070cc9e59b")) return kOverflow;
  hash_edge(case_digest.data(), 1, 2, 0, 0, edge);
  if (!digest_matches_hex(edge, 16, "2e5aed76eaf0270e31d51c6027870360")) return kOverflow;
  hash_essential(case_digest.data(), 0, baseline_field, death);
  if (!digest_matches_hex(death, 16, "52b54371e52653ac86315d8262321c7e")) return kOverflow;
  hash_interval(case_digest.data(), 0, baseline_field, edge, 10, 20, interval);
  if (!digest_matches_hex(interval, 16, "4d597be02e71f80571efc2b60495c56e")) return kOverflow;
  hash_interval(case_digest.data(), 0, baseline_field, death, 10, 65535, interval);
  if (!digest_matches_hex(interval, 16, "cf4e90a43b10c1316c240aa309e4cf17")) return kOverflow;
  hash_essential(case_digest.data(), 1, isotropic_field, death);
  if (!digest_matches_hex(death, 16, "9b7ee31a830d9fbf9c08dd1f25205fce")) return kOverflow;
  hash_interval(case_digest.data(), 1, isotropic_field, edge, 10, 20, interval);
  if (!digest_matches_hex(interval, 16, "1bbcb2248cdaadadaf40c12ebbd7535c")) return kOverflow;
  hash_interval(case_digest.data(), 1, isotropic_field, death, 10, 65535, interval);
  if (!digest_matches_hex(interval, 16, "0a1538fb562f41615668cf31c77f8f3e")) return kOverflow;

  const std::uint32_t support_count[1] = {1};
  const std::int64_t zero_moment[1] = {0};
  const std::uint8_t active[1] = {1};
  std::int64_t trace_q[1] = {0}, lambda1[1] = {0}, lambda2[1] = {0}, lambda3[1] = {0};
  std::uint16_t anisotropy[1] = {0}, filtration[1] = {65534};
  std::uint8_t result_code[1] = {0};
  const int status = ciip_v1_moments_to_field_masked(
      support_count, zero_moment, zero_moment, zero_moment, zero_moment, zero_moment,
      zero_moment, zero_moment, zero_moment, zero_moment, active, 1, trace_q, lambda1,
      lambda2, lambda3, anisotropy, filtration, result_code);
  if (status != kOk || result_code[0] != 0 || trace_q[0] != 274877906943LL ||
      lambda1[0] != 91625968981LL || lambda2[0] != 91625968981LL ||
      lambda3[0] != 91625968981LL || anisotropy[0] != 0 || filtration[0] != 65534) return kOverflow;
  return kOk;
}

}  // extern "C"
