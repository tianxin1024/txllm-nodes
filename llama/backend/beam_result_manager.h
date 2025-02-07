#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <map>

#include "backend/generator.h"

namespace generator {

class BeamHypothesis {
public:
    std::unordered_set<int> token_id_set;
    std::vector<std::pair<int, float>> top_logprobs;
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

}; // end of class BeamSearchResultManager

} // namespace generator
