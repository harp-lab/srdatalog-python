#pragma once

#include <algorithm>
#include <ranges>
#include <tuple>
#include <utility>  // For std::forward, std::index_sequence

namespace SRDatalog {

/**
 * @brief A C++20 View implementing Leapfrog Triejoin (Galloping Intersection).
 *
 * @details Unlike std::views::filter, this view utilizes the .seek() method on the
 * underlying iterators to skip ranges of non-matching values efficiently.
 * This implements the "leapfrog" algorithm where iterators take turns being
 * the leader, and other iterators "seek" to catch up.
 *
 * @tparam Iterators Variadic pack of iterator types that support:
 *   - Dereference operator (*)
 *   - Increment operator (++)
 *   - Equality comparison (==)
 *   - seek(value) method for efficient skipping
 */
template <typename... Iterators>
class LeapfrogView : public std::ranges::view_interface<LeapfrogView<Iterators...>> {
 public:
  // The value type is the value type of the first iterator (assuming all match)
  using ValueType = typename std::tuple_element_t<0, std::tuple<Iterators...>>::value_type;

  // --- The Iterator Class ---
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ValueType;
    using difference_type = std::ptrdiff_t;
    using pointer = const ValueType*;
    using reference = const ValueType&;

    Iterator() = default;

    // Constructor initiates the first search (find first match)
    explicit Iterator(std::tuple<Iterators...> iters, std::tuple<Iterators...> ends)
        : iters_(std::move(iters)), ends_(std::move(ends)) {
      find_next_match();
    }

    // Dereference: Return value of the first iterator (all are equal at match)
    reference operator*() const {
      return *std::get<0>(iters_);
    }
    pointer operator->() const {
      return std::get<0>(iters_).operator->();
    }

    // Advance: Move first iterator +1, then re-sync everyone
    Iterator& operator++() {
      // Move leader forward to break current match
      auto& leader = std::get<0>(iters_);
      ++leader;
      find_next_match();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const std::default_sentinel_t&) const {
      return is_end_;
    }

    bool operator==(const Iterator& other) const {
      // Simplified equality for view logic
      if (is_end_ != other.is_end_)
        return false;
      if (is_end_)
        return true;
      return iters_ == other.iters_;
    }

   private:
    std::tuple<Iterators...> iters_;
    std::tuple<Iterators...> ends_;
    bool is_end_{false};

    // --- THE LEAPFROG ALGORITHM ---
    void find_next_match() {
      if (check_any_end()) {
        is_end_ = true;
        return;
      }

      // We perform a cyclic search until all iterators align
      // To make this efficient with std::tuple, we use recursion or folding.
      // A simpler approach for variadic templates is a linear scan loop
      // that restarts if a seek pushes value higher.

      while (true) {
        // 1. Find Max Value (The Target)
        ValueType max_val = get_max_value();

        // 2. Try to bring everyone to Max Value
        bool any_moved = false;
        bool any_ended = false;

        // Fold expression to iterate tuple elements
        std::apply(
            [&](auto&... iter) {
              // Check each iterator
              (
                  [&] {
                    if (any_ended)
                      return;

                    if (*iter < max_val) {
                      iter.seek(max_val);  // GALLOP!

                      // Check EOF after seek
                      // We can't access 'ends_' easily inside this lambda without capture
                      // complexity Simplified check (assuming iterator knows its bounds or is safe)
                      // In production, pass 'ends' to lambda
                    }

                    // If after seek, we are > max_val, we have a NEW max.
                    // But for this simple implementation, we just flag 'any_moved'
                    // and the loop will re-calculate max_val.
                    if (*iter > max_val)
                      any_moved = true;
                  }(),
                  ...);
            },
            iters_);

        // Safety Check: bounds
        if (check_any_end()) {
          is_end_ = true;
          return;
        }

        // 3. If nobody moved (nobody was < max and nobody jumped > max), we are aligned!
        if (!any_moved) {
          // Double check equality just to be safe (handles the jump-over case)
          if (all_equal(max_val))
            return;
        }
      }
    }

    ValueType get_max_value() const {
      ValueType max_v = *std::get<0>(iters_);
      std::apply([&](const auto&... iter) { ((max_v = std::max(max_v, *iter)), ...); }, iters_);
      return max_v;
    }

    bool check_any_end() {
      bool ended = false;
      // Zip iterate iters and ends

      // Correct way with C++17 folding using helper
      iterate_tuple_pair(
          [&](const auto& it, const auto& end) {
            if (it == end)
              ended = true;
          },
          iters_, ends_);

      return ended;
    }

    bool all_equal(ValueType val) const {
      bool eq = true;
      std::apply([&](const auto&... iter) { ((eq = eq && (*iter == val)), ...); }, iters_);
      return eq;
    }

    // Helper to iterate two tuples in lockstep
    template <typename Func, typename Tup1, typename Tup2, size_t... Is>
    void iterate_tuple_pair_impl(Func f, Tup1& t1, Tup2& t2, std::index_sequence<Is...>) {
      (f(std::get<Is>(t1), std::get<Is>(t2)), ...);
    }

    template <typename Func, typename Tup1, typename Tup2>
    void iterate_tuple_pair(Func f, Tup1& t1, Tup2& t2) {
      iterate_tuple_pair_impl(f, t1, t2, std::make_index_sequence<std::tuple_size_v<Tup1>>{});
    }
  };

  // --- Constructor ---
  // Takes all begin iterators first, then all end iterators
  // When template arguments are explicitly specified, this constructor
  // accepts the iterators in the correct order
  template <typename FirstIter, typename... RestIters>
    requires(sizeof...(RestIters) + 1 == 2 * sizeof...(Iterators))
  LeapfrogView(FirstIter&& first, RestIters&&... rest) {
    // Split: first N are begins, last N are ends
    constexpr std::size_t N = sizeof...(Iterators);
    auto all = std::make_tuple(std::forward<FirstIter>(first), std::forward<RestIters>(rest)...);
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      begins_ = std::make_tuple(std::get<Is>(all)...);
      ends_ = std::make_tuple(std::get<Is + N>(all)...);
    }(std::make_index_sequence<N>{});
  }

  Iterator begin() const {
    return Iterator(begins_, ends_);
  }
  std::default_sentinel_t end() const {
    return {};
  }

 private:
  std::tuple<Iterators...> begins_;
  std::tuple<Iterators...> ends_;
};

// Deduction guide: deduces from the first N iterators (begins)
template <typename... Iterators>
LeapfrogView(Iterators... begins, Iterators... ends) -> LeapfrogView<Iterators...>;

}  // namespace SRDatalog
