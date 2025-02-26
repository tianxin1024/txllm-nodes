#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <map>

#include "backend/generator.h"

namespace generator {

class BeamHypothesis {
public:
    std::unordered_set<int> token_id_set;
    std::vector<std::pair<int, float>> top_logprobs;
    void add_token(int token_id, float accumulated_log_prob) {
        token_id_set.insert(token_id);
    }

    std::vector<std::map<int, float>> get_top_logprobs(size_t num_top) {
        std::vector<std::map<int, float>> maps(top_logprobs.size() / num_top);
        for (int i = 0; i < maps.size(); ++i) {
            for (int k = 0; k < num_top; ++k) {
                size_t index = i * num_top + k;
                maps[i].emplace(top_logprobs[index]);
            }
        }
        return maps;
    }

}; // end of class BeamHypothesis

template <typename BeamResult>
class BeamSearchResultManager {
    int num_results;
    std::vector<std::vector<BeamResult>> result_list;
    std::vector<std::vector<std::map<BeamResult, float>>> top_logprobs;
    std::vector<std::vector<std::map<BeamResult, float>>> logprobs;
    std::vector<float> cumulative_logprobs;
    std::vector<float> result_score;
    int current_results;
    float min_score{};

public:
    explicit BeamSearchResultManager(int num_results) :
        num_results(num_results), current_results(0), min_score(1e10) {
        result_list.resize(num_results);
        result_score.resize(num_results);
        top_logprobs.resize(num_results);
        cumulative_logprobs.resize(num_results);
    }

    BeamSearchResultManager(const BeamSearchResultManager &other) {
        num_results = other.num_results;
        result_list = other.result_list;
        result_score = other.result_score;
        top_logprobs = other.top_logprobs;
        logprobs = other.logprobs;
        cumulative_logprobs = other.cumulative_logprobs;
        current_results = other.current_results;
        min_score = other.min_score;
    }

    BeamSearchResultManager &operator=(BeamSearchResultManager &&other) {
        // use swqp and reset() to reserve other's vector's capacity
        num_results = other.num_results;
        result_list.swap(other.result_list);
        result_score.swap(other.result_score);
        top_logprobs.swap(other.top_logprobs);
        logprobs.swap(other.logprobs);
        cumulative_logprobs.swap(other.cumulative_logprobs);
        current_results = other.current_results;
        min_score = other.min_score;
        other.reset(0);
        return *this;
    }

    void reset(int new_num_results) {
        this->num_results = new_num_results;
        current_results = 0;
        min_score = 1e10;
        result_list.clear();
        result_score.clear();
        top_logprobs.resize(num_results);
        logprobs.resize(num_results);
        cumulative_logprobs.resize(num_results);
        result_list.resize(num_results);
        result_score.resize(num_results);
    }

    int get_current_results() const {
        return current_results;
    }

    std::vector<SearchResult> get_search_results(size_t topk = 1) {
        std::vector<std::pair<float, int>> score_ids;
        score_ids.reserve(current_results);
        for (int i = 0; i < current_results; i++) {
            score_ids.emplace_back(result_score[i], i);
        }
        std::sort(score_ids.begin(),
                  score_ids.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                      return a.first > b.first;
                  });
        if (topk == 0) {
            topk = 1;
        }
        std::vector<SearchResult> results;
        size_t results_num = topk > current_results ? current_results : topk;
        results.reserve(results_num);
        for (size_t i = 0; i < results_num; i++) {
            size_t index = score_ids[i].second;
            SearchResult result;
            result.set_tokens(std::move(result_list[index]));
            result.set_logprobs(std::move(logprobs[index]));
            result.set_top_logprobs(std::move(top_logprobs[index]));
            result.cumulative_logprob = cumulative_logprobs[index];
            result.score = result_score[index];
            results.emplace_back(result);
        }
        return results;
    }

    bool full() {
        return current_results >= num_results;
    }

    bool accept_score(float score) {
        return !full() || score >= min_score;
    }

    int add_result(const std::vector<BeamResult> &result,
                   const std::vector<std::map<BeamResult, float>> &logprobs,
                   float cumulative_logprob,
                   float score) {
        if (!full()) {
            result_list[current_results] = result;
            result_score[current_results] = score;
            this->logprobs[current_results] = logprobs;
            cumulative_logprobs[current_results] = cumulative_logprob;
            current_results++;
            if (score < min_score) {
                min_score = score;
            }
            return current_results - 1;
        } else {
            if (score < min_score) {
                return -1;
            }
            // result full, replace the result which score is lowest
            int min_idx = get_min_score_idx();
            if (score > result_score[min_idx]) {
                result_list[min_idx] = result;
                result_score[min_idx] = score;
                this->logprobs[min_idx] = logprobs;
                cumulative_logprobs[min_idx] = cumulative_logprob;
            }
            min_score = result_score[get_min_score_idx()];
            return min_idx;
        }
    }

    void set_top_logprobs(int i, std::vector<std::map<int, float>> &&x) {
        top_logprobs[i] = x;
    }

    int get_min_score_idx() const {
        int min_idx = 0;
        for (int i = 1; i < num_results; i++) {
            if (result_score[i] < result_score[min_idx]) {
                min_idx = i;
            }
        }
        return min_idx;
    }

}; // end of class BeamSearchResultManager

} // namespace generator
