#pragma once
#include "bmengine/core/core.h"
#include <memory>

namespace bmengine {

namespace storage {

class BMENGINE_EXPORT LocalStorage : public core::Storage {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    LocalStorage();
    ~LocalStorage();
    void fetch_parameter(core::ParameterData& data) override;
    size_t used_memory() const override;

    int add_file(const std::string& filename, const std::string& prefix);
    int load_from_file(
        const core::Context& ctx,
        const std::string& filename,
        const std::string& prefix,
        core::Layer* layer);

    std::vector<std::string> parameters() const;

    static std::map<std::string, const core::Tensor> load_file(
        const std::string& filename, const std::string& prefix);
};

} // namespace storage

} // namespace bmengine
