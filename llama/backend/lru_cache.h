#pragma once

#include <list>
#include <unordered_map>
#include <vector>

namespace utils {

struct IntVecHasher {
    size_t operator()(const std::vector<int> &vec) const {
        uint32_t seed = uint32_t(vec.size());
        for (auto &i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return size_t(seed);
    }
};

template <class K, class V, class HASH = std::hash<K>>
class LRUCache {
private:
    std::list<std::pair<K, V>> item_list;
    std::unordered_map<K, decltype(item_list.begin()), HASH> item_map;
    size_t cache_size_;

public:
    explicit LRUCache(int cache_size) :
        cache_size_(cache_size) {
    }

    bool get(const K &key, V &v) {
        auto it = item_map.find(key);
        if (it == item_map.end()) return false;
        item_list.splice(item_list.begin(), item_list, it->second);
        v = it->second->second;
        return true;
    }

}; // end of class LRUCache

} // namespace utils
