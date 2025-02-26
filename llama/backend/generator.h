#pragma once
#include <vector>
#include <map>
#include <iostream>

namespace generator {

struct SearchResult {
public:
    std::vector<int32_t> tokens;
    std::vector<std::map<int32_t, float>> top_logprobs; // top logprobs of each steps
    std::vector<std::map<int32_t, float>> logprobs;     // logprobs with candidates.
    float cumulative_logprob{0};                        // raw cumulative logprob
    float score{0};                                     // cumulative logprob with penalty applied

    void set_tokens(std::vector<int32_t> &&tokens) {
        this->tokens = tokens;
    }
    void set_logprobs(std::vector<std::map<int32_t, float>> &&logprobs) {
        this->logprobs = std::move(logprobs);
    }

    void set_top_logprobs(std::vector<std::map<int32_t, float>> &&logprobs) {
        this->top_logprobs = std::move(logprobs);
    }

    size_t tokens_num() const {
        return tokens.size();
    }
};

struct StreamResult : public SearchResult {
public:
    int update_flag;
    int step;

    void clear() {
        tokens.clear();
        score = -1e20;
    }

    void append(int token) {
        tokens.push_back(token);
        step++;
        update_flag = 1;
    }

    void update(std::vector<int32_t> &&tmp_res, int step1) {
        tokens = tmp_res;
        step = step1;
        update_flag = 2;
    }

}; // end of struct StreamResult

struct SearchResults {
public:
    std::vector<SearchResult> results;
    StreamResult stream;
    std::string stop_reason;
    long first_token_delay_ms{0};

    SearchResults() = default;
    SearchResults(const SearchResults &) = default;
    SearchResults &operator=(const SearchResults &other) = default;
    SearchResults &operator=(SearchResults &&other) = default;

}; // end of struct SearchResults

} // namespace generator
