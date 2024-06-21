/* Copyright 2022 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mappers/logging_wrapper.h"
#include "mappers/default_mapper.h"

#include "dsl_mapper.h"

#include "compiler/y.tab.c"
#include "compiler/lex.yy.c"

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <chrono>
// #include <mutex>

// using namespace Legion;
// using namespace Legion::Mapping;

// #define DEBUG_REGION_PLACEMENT
// #define DEBUG_MEMORY_COLLECT
// #define DEBUG_COMMAND_LINE

// static Logger log_mapper("nsmapper");
legion_equality_kind_t myop2legion(BinOpEnum myop);
std::string processor_kind_to_string(Processor::Kind kind);
std::string memory_kind_to_string(Memory::Kind kind);

namespace Legion
{
  namespace Internal
  {
    class UserShardingFunctor : public ShardingFunctor
    {
    private:
      std::string taskname;
      Tree2Legion tree;

    public:
      UserShardingFunctor(std::string takename_, const Tree2Legion &tree_);
      UserShardingFunctor(const UserShardingFunctor &rhs);
      virtual ~UserShardingFunctor(void);

    public:
      UserShardingFunctor &operator=(const UserShardingFunctor &rhs);

    public:
      virtual ShardID shard(const DomainPoint &point,
                            const Legion::Domain &full_space,
                            const size_t total_shards);
    };
  }
}

class NSMapper : public DefaultMapper
{
public:
  NSMapper(MapperRuntime *rt, Machine machine, Processor local, const char *mapper_name, bool first);

public:
  static std::string get_policy_file();
  static void parse_policy_file(const std::string &policy_file);
  static void register_user_sharding_functors(Runtime *runtime);
  void build_proc_idx_cache();

protected:
  std::vector<std::vector<Processor>> all_gpus;
  std::vector<std::vector<Processor>> all_cpus;
  std::vector<std::vector<Processor>> all_ios;
  std::vector<std::vector<Processor>> all_procsets;
  std::vector<std::vector<Processor>> all_omps;
  std::vector<std::vector<Processor>> all_pys;

private:
  Processor select_initial_processor_by_kind(const Task &task, Processor::Kind kind);
  bool validate_processor_mapping(MapperContext ctx, const Task &task, Processor proc, bool strict = true);
  template <typename Handle>
  void maybe_append_handle_name(const MapperContext ctx,
                                const Handle &handle,
                                std::vector<std::string> &names);
  void get_handle_names(const MapperContext ctx,
                        const RegionRequirement &req,
                        std::vector<std::string> &names);
  std::map<Legion::LogicalRegion, std::vector<std::string>> Region2Names;
  // Backpressure: https://github.com/StanfordLegion/legion/blob/stable/examples/mapper_backpressure/backpressure.cc
  // InFlightTask represents a task currently being executed.
  struct InFlightTask
  {
    // Unique identifier of the task instance.
    std::pair<Legion::Domain, size_t> id; // for index launch
    Legion::UniqueID id2;                 // for non-index launch task
    // An event that will be triggered when the task finishes.
    Legion::Mapping::MapperEvent event;
    // A clock measurement from when the task was scheduled.
    std::chrono::high_resolution_clock::time_point schedTime;
  };
  // backPressureQueue maintains state for each processor about how many
  // tasks that are marked to be backpressured are executing on the processor.
  std::map<Legion::Processor, std::deque<InFlightTask>> backPressureQueue;

public:
  virtual bool dsl_default_create_custom_instances(MapperContext ctx,
                                                   std::string task_name,
                                                   Processor target_proc, Memory target_memory,
                                                   const RegionRequirement &req, unsigned index,
                                                   std::set<FieldID> &needed_fields,
                                                   const TaskLayoutConstraintSet &layout_constraints,
                                                   bool needs_field_constraint_check,
                                                   std::vector<PhysicalInstance> &instances,
                                                   size_t *footprint /*= NULL*/);
  virtual bool dsl_default_make_instance(MapperContext ctx,
                                         Memory target_memory, const LayoutConstraintSet &constraints,
                                         PhysicalInstance &result, MappingKind kind, bool force_new, bool meets,
                                         const RegionRequirement &req, size_t *footprint);
  virtual void dsl_default_policy_select_constraints(MapperContext ctx,
                                                     std::string taskname, unsigned idx,
                                                     LayoutConstraintSet &constraints, Memory target_memory,
                                                     const RegionRequirement &req);
  LayoutConstraintID dsl_default_policy_select_layout_constraints(MapperContext ctx,
                                                                  std::string task_name,
                                                                  unsigned idx,
                                                                  Memory target_memory,
                                                                  const RegionRequirement &req,
                                                                  MappingKind mapping_kind,
                                                                  bool needs_field_constraint_check,
                                                                  bool &force_new_instances);
  virtual Processor dsl_default_policy_select_initial_processor(MapperContext ctx,
                                                                const Task &task);
  virtual void dsl_default_policy_select_target_processors(MapperContext ctx,
                                                           const Task &task,
                                                           std::vector<Processor> &target_procs);
  Memory dsl_default_policy_select_target_memory(MapperContext ctx,
                                                 std::string task_name,
                                                 Processor target_proc,
                                                 unsigned idx,
                                                 const RegionRequirement &req,
                                                 MemoryConstraint mc);
  virtual Legion::LogicalRegion dsl_default_policy_select_instance_region(MapperContext ctx,
                                                                          Memory target_memory,
                                                                          const RegionRequirement &req,
                                                                          const LayoutConstraintSet &constraints,
                                                                          bool force_new_instances,
                                                                          bool meets_constraints);
  virtual void map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output);
  virtual void map_replicate_task(const MapperContext ctx,
                                  const Task &task,
                                  const MapTaskInput &input,
                                  const MapTaskOutput &def_output,
                                  MapReplicateTaskOutput &output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Task &task,
                                       const SelectShardingFunctorInput &input,
                                       SelectShardingFunctorOutput &output);
  virtual void slice_task(const MapperContext ctx,
                          const Task &task,
                          const SliceTaskInput &input,
                          SliceTaskOutput &output) override;
  virtual void default_policy_select_sources(MapperContext ctx,
                                             const PhysicalInstance &target,
                                             const std::vector<PhysicalInstance> &sources,
                                             std::deque<PhysicalInstance> &ranking) override;
  virtual void select_task_options(const MapperContext ctx,
                                   const Task &task,
                                   TaskOptions &output) override;
  void dsl_default_remove_cached_task(MapperContext ctx,
                                      VariantID chosen_variant,
                                      unsigned long long task_hash,
                                      const std::pair<TaskID, Processor> &cache_key,
                                      const std::vector<std::vector<PhysicalInstance>> &post_filter);
  MapperSyncModel get_mapper_sync_model() const override;
  void report_profiling(const Legion::Mapping::MapperContext ctx,
                        const Legion::Task &task,
                        const TaskProfilingInfo &input) override;

  void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput &input,
                           SelectMappingOutput &output) override;
  Processor idx_to_proc(unsigned proc_idx, const Processor::Kind proc_kind) const;

protected:
  void map_task_post_function(const MapperContext &ctx,
                              const Task &task,
                              const std::string &task_name,
                              MapTaskOutput &output);
  Memory query_best_memory_for_proc(const Processor &proc,
                                    const Memory::Kind &mem_target_kind);
  void dsl_slice_task(const Task &task,
                      const std::vector<Processor> &local_procs,
                      const std::vector<std::vector<Processor>> &all_procs,
                      const SliceTaskInput &input,
                      SliceTaskOutput &output);
  template <int DIM>
  void dsl_decompose_points(std::vector<int> &index_launch_space,
                            const DomainT<DIM, coord_t> &point_space,
                            const std::vector<Processor> &targets_local,
                            const std::vector<std::vector<Processor>> &targets_all,
                            bool recurse, bool stealable,
                            std::vector<TaskSlice> &slices,
                            std::string taskname,
                            bool control_replicated);

private:
  std::unordered_map<TaskID, Processor::Kind> cached_task_policies;
  std::map<std::pair<Legion::Processor, Memory::Kind>, Legion::Memory> cached_affinity_proc2mem;
  std::map<std::pair<TaskID, Processor>, std::list<CachedTaskMapping>> dsl_cached_task_mappings;

public:
  std::map<Processor, int> proc_idx_cache;
  static Tree2Legion tree_result;
  static std::unordered_map<std::string, ShardingID> task2sid;
  static bool backpressure;
  static bool untrackValidRegions;
  static bool use_semantic_name;
  static bool select_source_by_bandwidth;
  static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs);
};

Tree2Legion NSMapper::tree_result;
std::unordered_map<std::string, ShardingID> NSMapper::task2sid;
bool NSMapper::backpressure;
bool NSMapper::untrackValidRegions;
bool NSMapper::use_semantic_name;
bool NSMapper::select_source_by_bandwidth;

std::string NSMapper::get_policy_file()
{
  auto args = Runtime::get_input_args();
  for (auto idx = 0; idx < args.argc; ++idx)
  {
    if (strcmp(args.argv[idx], "-mapping") == 0)
    {
      if (idx + 1 >= args.argc)
        break;
      return args.argv[idx + 1];
    }
  }
  printf("Policy file is missing\n");
  exit(-1);
}

inline std::string processor_kind_to_string(Processor::Kind kind)
{
  switch (kind)
  {
  case Processor::LOC_PROC:
    return "CPU";
  case Processor::TOC_PROC:
    return "GPU";
  case Processor::IO_PROC:
    return "IO";
  case Processor::PY_PROC:
    return "PY";
  case Processor::PROC_SET:
    return "PROC";
  case Processor::OMP_PROC:
    return "OMP";
  default:
  {
    assert(false);
    return "Unknown Kind";
  }
  }
}

std::string memory_kind_to_string(Memory::Kind kind)
{
  switch (kind)
  {
  case Memory::SYSTEM_MEM:
    return "SYSMEM";
  case Memory::GPU_FB_MEM:
    return "FBMEM";
  case Memory::REGDMA_MEM:
    return "RDMEM";
  case Memory::Z_COPY_MEM:
    return "ZCMEM";
  case Memory::SOCKET_MEM:
    return "SOCKETMEM";
  default:
  {
    assert(false);
    return "Unknown Kind";
  }
  }
}

void NSMapper::register_user_sharding_functors(Runtime *runtime)
{
  int i = 1;
  for (auto v : tree_result.indextask2func)
  {
    runtime->register_sharding_functor(i, new Legion::Internal::UserShardingFunctor(v.first, tree_result));
    task2sid.insert({v.first, i});
    // log_mapper.debug("%s inserted", v.first.c_str());
    i += 1;
  }
}

void NSMapper::parse_policy_file(const std::string &policy_file)
{
  // log_mapper.debug("Policy file: %s", policy_file.c_str());
  tree_result = Tree2Legion(policy_file);
  tree_result.print();
}

inline Processor NSMapper::idx_to_proc(unsigned proc_idx, const Processor::Kind proc_kind) const
{
  switch (proc_kind)
  {
  case Processor::LOC_PROC:
  {
    assert(proc_idx < this->local_cpus.size());
    return this->local_cpus[proc_idx];
  }
  case Processor::TOC_PROC:
  {
    assert(proc_idx < this->local_gpus.size());
    return this->local_gpus[proc_idx];
  }
  case Processor::IO_PROC:
  {
    assert(proc_idx < this->local_ios.size());
    return this->local_ios[proc_idx];
  }
  case Processor::PY_PROC:
  {
    assert(proc_idx < this->local_pys.size());
    return this->local_pys[proc_idx];
  }
  case Processor::PROC_SET:
  {
    assert(proc_idx < this->local_procsets.size());
    return this->local_procsets[proc_idx];
  }
  case Processor::OMP_PROC:
  {
    assert(proc_idx < this->local_omps.size());
    return this->local_omps[proc_idx];
  }
  default:
  {
    assert(false);
  }
  }
  assert(false);
  return this->local_cpus[0];
}

void NSMapper::build_proc_idx_cache()
{
  for (size_t i = 0; i < this->local_cpus.size(); i++)
  {
    this->proc_idx_cache.insert({this->local_cpus[i], i});
  }
  for (size_t i = 0; i < this->local_gpus.size(); i++)
  {
    this->proc_idx_cache.insert({this->local_gpus[i], i});
  }
  for (size_t i = 0; i < this->local_omps.size(); i++)
  {
    this->proc_idx_cache.insert({this->local_omps[i], i});
  }
  // for (size_t i  = 0; i < this->local_ios.size(); i++)
  // {
  //   this->proc_idx_cache.insert({this->local_ios[i], i});
  // }
  // for (size_t i  = 0; i < this->local_pys.size(); i++)
  // {
  //   this->proc_idx_cache.insert({this->local_pys[i], i});
  // }
  // for (size_t i  = 0; i < this->local_procsets.size(); i++)
  // {
  //   this->proc_idx_cache.insert({this->local_procsets[i], i})
  // }
}

Processor NSMapper::select_initial_processor_by_kind(const Task &task, Processor::Kind kind)
{
  Processor result = local_cpus.front();
  switch (kind)
  {
  case Processor::LOC_PROC:
  {
    result = local_cpus.front();
    break;
  }
  case Processor::TOC_PROC:
  {
    result = !local_gpus.empty() ? local_gpus.front() : local_cpus.front();
    break;
  }
  case Processor::IO_PROC:
  {
    result = !local_ios.empty() ? local_ios.front() : local_cpus.front();
    break;
  }
  case Processor::PY_PROC:
  {
    result = !local_pys.empty() ? local_pys.front() : local_cpus.front();
    break;
  }
  case Processor::PROC_SET:
  {
    result = !local_procsets.empty() ? local_procsets.front() : local_cpus.front();
    break;
  }
  case Processor::OMP_PROC:
  {
    result = !local_omps.empty() ? local_omps.front() : local_cpus.front();
    break;
  }
  default:
  {
    assert(false);
  }
  }

  // auto kind_str = processor_kind_to_string(kind);
  // if (result.kind() != kind)
  // {
  //   log_mapper.warning(
  //     "Unsatisfiable policy: task %s requested %s, which does not exist",
  //     task.get_task_name(), kind_str.c_str());
  // }
  // else
  // {
  //   // log_mapper.debug(
  //   //   "Task %s is initially mapped to %s",
  //   //   task.get_task_name(), kind_str.c_str()
  //   // );
  // }
  return result;
}

bool NSMapper::validate_processor_mapping(MapperContext ctx, const Task &task, Processor proc, bool strict)
{
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants, proc.kind());
  if (variants.empty())
  {
    if (strict)
    {
      auto kind_str = processor_kind_to_string(proc.kind());
      printf("Invalid policy: task %s requested %s, but has no valid task variant for the kind",
             task.get_task_name(), kind_str.c_str());
      exit(-1);
    }
    else
    {
      return false;
    }
  }
  return true;
}

Processor NSMapper::dsl_default_policy_select_initial_processor(MapperContext ctx, const Task &task)
{
  {
    auto finder = cached_task_policies.find(task.task_id);
    if (finder != cached_task_policies.end())
    {
      auto result = select_initial_processor_by_kind(task, finder->second);
      validate_processor_mapping(ctx, task, result);
      // log_mapper.debug() << task.get_task_name() << " mapped by cache: " << processor_kind_to_string(result.kind()).c_str();
      return result;
    }
  }
  std::string task_name = task.get_task_name();
  {
    std::vector<Processor::Kind> proc_kind_vec;
    if (tree_result.task_policies.count(task_name) > 0)
    {
      proc_kind_vec = tree_result.task_policies.at(task_name);
    }
    else if (tree_result.task_policies.count("*") > 0)
    {
      proc_kind_vec = tree_result.task_policies.at("*");
    }
    for (size_t i = 0; i < proc_kind_vec.size(); i++)
    {
      auto result = select_initial_processor_by_kind(task, proc_kind_vec[i]);
      if (result.kind() != proc_kind_vec[i])
      {
        // log_mapper.debug("Mapping %s onto %s cannot satisfy, try next",
        // task_name.c_str(), processor_kind_to_string(proc_kind_vec[i]).c_str());
        continue;
      }
      // default policy validation should not be strict, allowing fallback
      bool success = validate_processor_mapping(ctx, task, result, false);
      if (success)
      {
        // log_mapper.debug() << task_name << " mapped to " << processor_kind_to_string(result.kind()).c_str();
        cached_task_policies[task.task_id] = result.kind();
        return result;
      }
      else
      {
        // log_mapper.debug("Mapping %s onto %s cannot satisfy with validation, try next",
        // task_name.c_str(), processor_kind_to_string(proc_kind_vec[i]).c_str());
      }
    }
  }
  // log_mapper.debug("%s falls back to the default policy", task_name.c_str());
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void NSMapper::dsl_default_policy_select_target_processors(MapperContext ctx,
                                                           const Task &task,
                                                           std::vector<Processor> &target_procs)
{
  if (tree_result.should_fall_back(task.get_task_name(), task.is_index_space, task.target_proc.kind()) == false)
  {
    std::vector<std::vector<int>> res;
    if (!task.is_index_space)
    {
      res = tree_result.runsingle(&task, this);
      // target_procs.push_back(task.orig_proc);
      // return;
    }
    else
    {
      res = tree_result.runindex(&task);
    }
    unsigned node_idx = (unsigned)res[0][0];
    assert(task.target_proc.address_space() == node_idx);
    for (size_t i = 0; i < res.size(); i++)
    {
      assert((unsigned)res[i][0] == node_idx); // must be on the same node
      // Todo: for round-robin semantic
      // We might also need to create physical instance for all the returned processors
      // Currently not implemented due to lack of motivating examples
      target_procs.push_back(idx_to_proc((unsigned)res[i][1], task.target_proc.kind()));
    }
  }
  else
  {
    DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
  }
}

Legion::LogicalRegion NSMapper::dsl_default_policy_select_instance_region(MapperContext ctx,
                                                                          Memory target_memory,
                                                                          const RegionRequirement &req,
                                                                          const LayoutConstraintSet &constraints,
                                                                          bool force_new_instances,
                                                                          bool meets_constraints)
{
  // Not invoked anywhere yet; can be used in map_task to replace default_policy_select_instance_region
  return req.region;
}

template <typename Handle>
void NSMapper::maybe_append_handle_name(const MapperContext ctx,
                                        const Handle &handle,
                                        std::vector<std::string> &names)
{
  const void *result = nullptr;
  size_t size = 0;
  if (runtime->retrieve_semantic_information(
          ctx, handle, LEGION_NAME_SEMANTIC_TAG, result, size, true, true))
    names.push_back(std::string(static_cast<const char *>(result)));
}

void NSMapper::get_handle_names(const MapperContext ctx,
                                const RegionRequirement &req,
                                std::vector<std::string> &names)
{
  if (Region2Names.count(req.region) > 0)
  {
    names = Region2Names.at(req.region);
  }
  maybe_append_handle_name(ctx, req.region, names);

  if (runtime->has_parent_logical_partition(ctx, req.region))
  {
    auto parent = runtime->get_parent_logical_partition(ctx, req.region);
    maybe_append_handle_name(ctx, parent, names);
  }

  if (req.region != req.parent)
    maybe_append_handle_name(ctx, req.parent, names);

  Region2Names.insert({req.region, names});
}

Memory NSMapper::query_best_memory_for_proc(const Processor &proc, const Memory::Kind &mem_target_kind)
{
  if (cached_affinity_proc2mem.count({proc, mem_target_kind}) > 0)
  {
    return cached_affinity_proc2mem.at({proc, mem_target_kind});
  }
  Machine::MemoryQuery visible_memories(machine);
  // visible_memories.local_address_space()
  visible_memories.same_address_space_as(proc)
      .only_kind(mem_target_kind);
  if (mem_target_kind != Memory::Z_COPY_MEM)
  {
    visible_memories.best_affinity_to(proc);
  }
  else
  {
    visible_memories.has_affinity_to(proc); // Z_COPY_MEM doesn't work using best_affinity_to
  }
  if (visible_memories.count() > 0)
  {
    Memory result = visible_memories.first();
    if (result.exists())
    {
      cached_affinity_proc2mem.insert({{proc, mem_target_kind}, result});
      return result;
    }
  }
  return Memory::NO_MEMORY;
}

void NSMapper::map_task_post_function(const MapperContext &ctx,
                                      const Task &task,
                                      const std::string &task_name,
                                      MapTaskOutput &output)
{
  if (NSMapper::untrackValidRegions && NSMapper::tree_result.memory_collect.size() > 0)
  {
    for (size_t i = 0; i < task.regions.size(); i++)
    {
      auto &rg = task.regions[i];
      if (rg.privilege == READ_ONLY)
      {
        if (use_semantic_name)
        {
          std::vector<std::string> path;
          get_handle_names(ctx, rg, path);
          if (tree_result.should_collect_memory(task_name, path))
          {
#ifdef DEBUG_MEMORY_COLLECT
            std::cout << "task " << task_name << " region " << i << " will be collected" << std::endl;
#endif
            output.untracked_valid_regions.insert(i);
          }
        }
        else
        {
          if (tree_result.should_collect_memory(task_name, {std::to_string(i)}))
          {
#ifdef DEBUG_MEMORY_COLLECT
            std::cout << "task " << task_name << " region " << i << " will be collected" << std::endl;
#endif
            output.untracked_valid_regions.insert(i);
          }
        }
      }
    }
  }

  if (NSMapper::backpressure && tree_result.query_max_instance(task_name) > 0)
  {
    output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
  }
  return;
}

void NSMapper::dsl_default_remove_cached_task(MapperContext ctx,
                                              VariantID chosen_variant, unsigned long long task_hash,
                                              const std::pair<TaskID, Processor> &cache_key,
                                              const std::vector<std::vector<PhysicalInstance>> &post_filter)
//--------------------------------------------------------------------------
{
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping>>::iterator
      finder = dsl_cached_task_mappings.find(cache_key);
  if (finder != dsl_cached_task_mappings.end())
  {
    // Keep a list of instances for which we need to downgrade
    // their garbage collection priorities since we are no
    // longer caching the results
    std::deque<PhysicalInstance> to_downgrade;
    for (std::list<CachedTaskMapping>::iterator it =
             finder->second.begin();
         it != finder->second.end(); it++)
    {
      if ((it->variant == chosen_variant) &&
          (it->task_hash == task_hash))
      {
        // Record all the instances for which we will need to
        // down grade their garbage collection priority
        for (unsigned idx1 = 0; (idx1 < it->mapping.size()) &&
                                (idx1 < post_filter.size());
             idx1++)
        {
          if (!it->mapping[idx1].empty())
          {
            if (!post_filter[idx1].empty())
            {
              // Still all the same
              if (post_filter[idx1].size() == it->mapping[idx1].size())
                continue;
              // See which ones are no longer in our set
              for (unsigned idx2 = 0;
                   idx2 < it->mapping[idx1].size(); idx2++)
              {
                PhysicalInstance current = it->mapping[idx1][idx2];
                bool still_valid = false;
                for (unsigned idx3 = 0;
                     idx3 < post_filter[idx1].size(); idx3++)
                {
                  if (current == post_filter[idx1][idx3])
                  {
                    still_valid = true;
                    break;
                  }
                }
                if (!still_valid)
                  to_downgrade.push_back(current);
              }
            }
            else
            {
              // if the chosen instances are empty, record them all
              to_downgrade.insert(to_downgrade.end(),
                                  it->mapping[idx1].begin(), it->mapping[idx1].end());
            }
          }
        }
        finder->second.erase(it);
        break;
      }
    }
    if (finder->second.empty())
      dsl_cached_task_mappings.erase(finder);
    if (!to_downgrade.empty())
    {
      for (std::deque<PhysicalInstance>::const_iterator it =
               to_downgrade.begin();
           it != to_downgrade.end(); it++)
      {
        if (it->is_external_instance())
          continue;
        runtime->set_garbage_collection_priority(ctx, *it, 0 /*priority*/);
      }
    }
  }
}

void NSMapper::map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output)
{
  std::string task_name = task.get_task_name();
  Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  VariantInfo chosen;
  if (input.shard_processor.exists())
  {
    const std::pair<TaskID,Processor::Kind> key(task.task_id, input.shard_processor.kind());
    std::map<std::pair<TaskID,Processor::Kind>,VariantInfo>::const_iterator finder = preferred_variants.find(key);
    if (finder == preferred_variants.end())
    {
      chosen.variant = input.shard_variant;
      chosen.proc_kind = input.shard_processor.kind();
      chosen.tight_bound = true;
      chosen.is_inner = runtime->is_inner_variant(ctx, task.task_id, input.shard_variant);
      chosen.is_replicable = true;
      preferred_variants.emplace(std::make_pair(key, chosen));
    }
    else
    {
      chosen = finder->second;
    }
  }
  else
  {
    chosen = DefaultMapper::default_find_preferred_variant(task, ctx, true /*needs tight bound*/, true /*cache*/, target_kind);
  }
  output.chosen_variant = chosen.variant;
  output.task_priority = 0;
  output.postmap_task = false;
  // Figure out our target processors
  if (input.shard_processor.exists())
  {
    output.target_procs.resize(1, input.shard_processor);
  }
  else
  {
    dsl_default_policy_select_target_processors(ctx, task, output.target_procs);
  }
  Processor target_proc = output.target_procs[0];
  // See if we have an inner variant, if we do virtually map all the regions
  // We don't even both caching these since they are so simple
  if (chosen.is_inner)
  {
    // Check to see if we have any relaxed coherence modes in which
    // case we can no longer do virtual mappings so we'll fall through
    bool has_relaxed_coherence = false;
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if (task.regions[idx].prop != LEGION_EXCLUSIVE)
      {
        has_relaxed_coherence = true;
        break;
      }
    }
    if (!has_relaxed_coherence)
    {
      std::vector<unsigned> reduction_indexes;
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        // As long as this isn't a reduction-only region requirement
        // we will do a virtual mapping, for reduction-only instances
        // we will actually make a physical instance because the runtime
        // doesn't allow virtual mappings for reduction-only privileges
        if (task.regions[idx].privilege == LEGION_REDUCE)
          reduction_indexes.push_back(idx);
        else
          output.chosen_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
      }
      if (!reduction_indexes.empty())
      {
        const TaskLayoutConstraintSet &layout_constraints =
            runtime->find_task_layout_constraints(ctx,
                                                  task.task_id, output.chosen_variant);
        for (std::vector<unsigned>::const_iterator it =
                 reduction_indexes.begin();
             it !=
             reduction_indexes.end();
             it++)
        {
          MemoryConstraint mem_constraint =
              DefaultMapper::find_memory_constraint(ctx, task, output.chosen_variant, *it);
          Memory target_memory = dsl_default_policy_select_target_memory(ctx,
                                                                         task_name,
                                                                         target_proc,
                                                                         *it,
                                                                         task.regions[*it],
                                                                         mem_constraint);
          std::set<FieldID> copy = task.regions[*it].privilege_fields;
          size_t footprint;
          if (!dsl_default_create_custom_instances(ctx, task_name, target_proc,
                                                   target_memory, task.regions[*it], *it, copy,
                                                   layout_constraints, false /*needs constraint check*/,
                                                   output.chosen_instances[*it], &footprint))
          {
            DefaultMapper::default_report_failed_instance_creation(task, *it,
                                                                   target_proc, target_memory, footprint);
          }
        }
      }
      map_task_post_function(ctx, task, task_name, output);
      return;
    }
  }
  // Should we cache this task?
  CachedMappingPolicy cache_policy =
      DefaultMapper::default_policy_select_task_cache_policy(ctx, task);

  // First, let's see if we've cached a result of this task mapping
  const unsigned long long task_hash = DefaultMapper::compute_task_hash(task);
  std::pair<TaskID, Processor> cache_key(task.task_id, target_proc);
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping>>::const_iterator
      finder = dsl_cached_task_mappings.find(cache_key);
  // This flag says whether we need to recheck the field constraints,
  // possibly because a new field was allocated in a region, so our old
  // cached physical instance(s) is(are) no longer valid
  bool needs_field_constraint_check = false;
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE && finder != dsl_cached_task_mappings.end())
  {
    bool found = false;
    // Iterate through and see if we can find one with our variant and hash
    for (std::list<CachedTaskMapping>::const_iterator it =
             finder->second.begin();
         it != finder->second.end(); it++)
    {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash))
      {
        // Have to copy it before we do the external call which
        // might invalidate our iterator
        output.chosen_instances = it->mapping;
        output.output_targets = it->output_targets;
        output.output_constraints = it->output_constraints;
        found = true;
        break;
      }
    }
    if (found)
    {
      // See if we can acquire these instances still
      if (runtime->acquire_and_filter_instances(ctx,
                                                output.chosen_instances))
      {
        map_task_post_function(ctx, task, task_name, output);
        return;
      }
      // We need to check the constraints here because we had a
      // prior mapping and it failed, which may be the result
      // of a change in the allocated fields of a field space
      needs_field_constraint_check = true;
      // If some of them were deleted, go back and remove this entry
      // Have to renew our iterators since they might have been
      // invalidated during the 'acquire_and_filter_instances' call
      dsl_default_remove_cached_task(ctx, output.chosen_variant,
                                     task_hash, cache_key, output.chosen_instances);
    }
  }
  // We didn't find a cached version of the mapping so we need to
  // do a full mapping, we already know what variant we want to use
  // so let's use one of the acceleration functions to figure out
  // which instances still need to be mapped.
  std::vector<std::set<FieldID>> missing_fields(task.regions.size());
  runtime->filter_instances(ctx, task, output.chosen_variant,
                            output.chosen_instances, missing_fields);
  // Track which regions have already been mapped
  std::vector<bool> done_regions(task.regions.size(), false);
  if (!input.premapped_regions.empty())
  {
    for (std::vector<unsigned>::const_iterator it = input.premapped_regions.begin();
         it != input.premapped_regions.end(); it++)
    {
      done_regions[*it] = true;
    }
  }
  const TaskLayoutConstraintSet &layout_constraints =
      runtime->find_task_layout_constraints(ctx,
                                            task.task_id, output.chosen_variant);
  // Now we need to go through and make instances for any of our
  // regions which do not have space for certain fields
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
  {
    if (done_regions[idx])
      continue;
    // Skip any empty regions
    if ((task.regions[idx].privilege == LEGION_NO_ACCESS) ||
        (task.regions[idx].privilege_fields.empty()) ||
        missing_fields[idx].empty())
      continue;
    // See if this is a reduction
    MemoryConstraint mem_constraint =
        DefaultMapper::find_memory_constraint(ctx, task, output.chosen_variant, idx);
    Memory target_memory = dsl_default_policy_select_target_memory(ctx,
                                                                   task_name,
                                                                   target_proc,
                                                                   idx,
                                                                   task.regions[idx],
                                                                   mem_constraint);
    if (task.regions[idx].privilege == LEGION_REDUCE)
    {
      size_t footprint;
      if (!dsl_default_create_custom_instances(ctx, task_name, target_proc,
                                               target_memory, task.regions[idx], idx, missing_fields[idx],
                                               layout_constraints, needs_field_constraint_check,
                                               output.chosen_instances[idx], &footprint))
      {
        DefaultMapper::default_report_failed_instance_creation(task, idx,
                                                               target_proc, target_memory, footprint);
      }
      continue;
    }
    // Did the application request a virtual mapping for this requirement with task's tag?
    // Did the user request a Virtual instance in the DSL?
    if (((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) != 0) || (target_memory == Memory::NO_MEMORY))
    {
      PhysicalInstance virt_inst = PhysicalInstance::get_virtual_instance();
      output.chosen_instances[idx].push_back(virt_inst);
      continue;
    }
    // Check to see if any of the valid instances satisfy this requirement
    {
      std::vector<PhysicalInstance> valid_instances;

      for (std::vector<PhysicalInstance>::const_iterator
               it = input.valid_instances[idx].begin(),
               ie = input.valid_instances[idx].end();
           it != ie; ++it)
      {
        if (it->get_location() == target_memory)
          valid_instances.push_back(*it);
      }

      std::set<FieldID> valid_missing_fields;
      runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                                valid_instances, valid_missing_fields);

#ifndef NDEBUG
      bool check =
#endif
          runtime->acquire_and_filter_instances(ctx, valid_instances);
      assert(check);

      output.chosen_instances[idx] = valid_instances;
      missing_fields[idx] = valid_missing_fields;

      if (missing_fields[idx].empty())
        continue;
    }
    // Otherwise make normal instances for the given region
    size_t footprint;
    if (!dsl_default_create_custom_instances(ctx, task_name, target_proc,
                                             target_memory, task.regions[idx], idx, missing_fields[idx],
                                             layout_constraints, needs_field_constraint_check,
                                             output.chosen_instances[idx], &footprint))
    {
      DefaultMapper::default_report_failed_instance_creation(task, idx,
                                                             target_proc, target_memory, footprint);
    }
  }

  // Finally we set a target memory for output instances
  Memory target_memory =
      DefaultMapper::default_policy_select_output_target(ctx, task.target_proc);
  for (unsigned i = 0; i < task.output_regions.size(); ++i)
  {
    output.output_targets[i] = target_memory;
    DefaultMapper::default_policy_select_output_constraints(
        task, output.output_constraints[i], task.output_regions[i]);
  }

  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE)
  {
    // Now that we are done, let's cache the result so we can use it later
    std::list<CachedTaskMapping> &map_list = dsl_cached_task_mappings[cache_key];
    map_list.push_back(CachedTaskMapping());
    CachedTaskMapping &cached_result = map_list.back();
    cached_result.task_hash = task_hash;
    cached_result.variant = output.chosen_variant;
    cached_result.mapping = output.chosen_instances;
    cached_result.output_targets = output.output_targets;
    cached_result.output_constraints = output.output_constraints;
  }
  map_task_post_function(ctx, task, task_name, output);
}

Memory NSMapper::dsl_default_policy_select_target_memory(MapperContext ctx,
                                                         std::string task_name,
                                                         Processor target_proc,
                                                         unsigned idx,
                                                         const RegionRequirement &req,
                                                         MemoryConstraint mc)
{
  std::vector<Memory::Kind> memory_list;
  if (use_semantic_name)
  {
#ifdef DEBUG_REGION_PLACEMENT
    printf("use_semantic_name\n");
#endif
    std::vector<std::string> path;
    get_handle_names(ctx, req, path);
    // log_mapper.debug() << "found_policy = false; path.size() = " << path.size(); // use index for regent
    memory_list = tree_result.query_memory_list(task_name, path, target_proc.kind());
#ifdef DEBUG_REGION_PLACEMENT
    for (auto i = 0; i < path.size(); i++)
    {
      printf("----start get_handle_names------\n");
      std::cout << path[i] << std::endl;
      printf("----end get_handle_names-----\n");
    }
#endif
  }
  else
  {
    memory_list = tree_result.query_memory_list(task_name, {std::to_string(idx)}, target_proc.kind());
  }
#ifdef DEBUG_REGION_PLACEMENT
  for (auto i = 0; i < memory_list.size(); i++)
  {
    printf("-----start query_memory_list ---\n");
    std::cout << memory_kind_to_string(memory_list[i]) << std::endl;
    printf("-----end query_memory_list ---\n");
  }
#endif
  for (auto &mem_kind : memory_list)
  {
    // log_mapper.debug() << "querying " << target_processor.id <<
    // " for memory " << memory_kind_to_string(mem_kind);
    if (mem_kind == Memory::NO_MEMKIND) // user request Virtual Instance
    {
      return Memory::NO_MEMORY;
    }
    Memory target_memory_try = query_best_memory_for_proc(target_proc, mem_kind);
    if (target_memory_try.exists())
    {
#ifdef DEBUG_REGION_PLACEMENT
      std::cout << "Placement:" << memory_kind_to_string(target_memory_try.kind()) << std::endl;
#endif
      return target_memory_try;
    }
  }
  // log_mapper.debug(
  // "Cannot find a policy for memory: region %u of task %s cannot be mapped, falling back to the default policy",
  // idx, task.get_task_name());
  return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
}

bool NSMapper::dsl_default_create_custom_instances(MapperContext ctx,
                                                   std::string task_name,
                                                   Processor target_proc, Memory target_memory,
                                                   const RegionRequirement &req, unsigned index,
                                                   std::set<FieldID> &needed_fields,
                                                   const TaskLayoutConstraintSet &layout_constraints,
                                                   bool needs_field_constraint_check,
                                                   std::vector<PhysicalInstance> &instances,
                                                   size_t *footprint /*= NULL*/)
//--------------------------------------------------------------------------
{
  // Special case for reduction instances, no point in checking
  // for existing ones and we also know that currently we can only
  // make a single instance for each field of a reduction
  if (req.privilege == LEGION_REDUCE)
  {
    // Iterate over the fields one by one for now, once Realm figures
    // out how to deal with reduction instances that contain
    bool force_new_instances = true; // always have to force new instances
    LayoutConstraintID our_layout_id =
        dsl_default_policy_select_layout_constraints(ctx, task_name, index, target_memory, req,
                                                     TASK_MAPPING, needs_field_constraint_check, force_new_instances);
    LayoutConstraintSet our_constraints =
        runtime->find_layout_constraints(ctx, our_layout_id);
    instances.resize(instances.size() + req.privilege_fields.size());
    unsigned idx = 0;
    for (std::set<FieldID>::const_iterator it =
             req.privilege_fields.begin();
         it !=
         req.privilege_fields.end();
         it++, idx++)
    {
      our_constraints.field_constraint.field_set.clear();
      our_constraints.field_constraint.field_set.push_back(*it);
      if (!dsl_default_make_instance(ctx, target_memory, our_constraints,
                                     instances[idx], TASK_MAPPING, force_new_instances,
                                     true /*meets*/, req, footprint))
        return false;
    }
    return true;
  }
  // Before we do anything else figure out our
  // constraints for any instances of this task, then we'll
  // see if these constraints conflict with or are satisfied by
  // any of the other constraints
  bool force_new_instances = false;
  LayoutConstraintID our_layout_id =
      dsl_default_policy_select_layout_constraints(ctx, task_name, index, target_memory, req,
                                                   TASK_MAPPING, needs_field_constraint_check, force_new_instances);
  const LayoutConstraintSet &our_constraints =
      runtime->find_layout_constraints(ctx, our_layout_id);
  for (std::multimap<unsigned, LayoutConstraintID>::const_iterator lay_it =
           layout_constraints.layouts.lower_bound(index);
       lay_it !=
       layout_constraints.layouts.upper_bound(index);
       lay_it++)
  {
    // Get the constraints
    const LayoutConstraintSet &index_constraints =
        runtime->find_layout_constraints(ctx, lay_it->second);
    std::vector<FieldID> overlapping_fields;
    const std::vector<FieldID> &constraint_fields =
        index_constraints.field_constraint.get_field_set();
    if (!constraint_fields.empty())
    {
      for (unsigned idx = 0; idx < constraint_fields.size(); idx++)
      {
        FieldID fid = constraint_fields[idx];
        std::set<FieldID>::iterator finder = needed_fields.find(fid);
        if (finder != needed_fields.end())
        {
          overlapping_fields.push_back(fid);
          // Remove from the needed fields since we're going to handle it
          needed_fields.erase(finder);
        }
      }
      // If we don't have any overlapping fields, then keep going
      if (overlapping_fields.empty())
        continue;
    }
    else // otherwise it applies to all the fields
    {
      overlapping_fields.insert(overlapping_fields.end(),
                                needed_fields.begin(), needed_fields.end());
      needed_fields.clear();
    }
    // Now figure out how to make an instance
    instances.resize(instances.size() + 1);
    // Check to see if these constraints conflict with our constraints
    // or whether they entail our mapper preferred constraints
    if (runtime->do_constraints_conflict(ctx, our_layout_id, lay_it->second) || runtime->do_constraints_entail(ctx, lay_it->second, our_layout_id))
    {
      // They conflict or they entail our constraints so we're just going
      // to make an instance using these constraints
      // Check to see if they have fields and if not constraints with fields
      if (constraint_fields.empty())
      {
        LayoutConstraintSet creation_constraints = index_constraints;
        dsl_default_policy_select_constraints(ctx, task_name, index, creation_constraints,
                                              target_memory, req);
        creation_constraints.add_constraint(
            FieldConstraint(overlapping_fields,
                            index_constraints.field_constraint.contiguous,
                            index_constraints.field_constraint.inorder));
        if (!dsl_default_make_instance(ctx, target_memory, creation_constraints,
                                       instances.back(), TASK_MAPPING, force_new_instances,
                                       true /*meets*/, req, footprint))
          return false;
      }
      else if (!dsl_default_make_instance(ctx, target_memory, index_constraints,
                                          instances.back(), TASK_MAPPING, force_new_instances,
                                          false /*meets*/, req, footprint))
        return false;
    }
    else
    {
      // These constraints don't do as much as we want but don't
      // conflict so make an instance with them and our constraints
      LayoutConstraintSet creation_constraints = index_constraints;
      dsl_default_policy_select_constraints(ctx, task_name, index, creation_constraints,
                                            target_memory, req);
      creation_constraints.add_constraint(
          FieldConstraint(overlapping_fields,
                          creation_constraints.field_constraint.contiguous ||
                              index_constraints.field_constraint.contiguous,
                          creation_constraints.field_constraint.inorder ||
                              index_constraints.field_constraint.inorder));
      if (!dsl_default_make_instance(ctx, target_memory, creation_constraints,
                                     instances.back(), TASK_MAPPING, force_new_instances,
                                     true /*meets*/, req, footprint))
        return false;
    }
  }
  // If we don't have anymore needed fields, we are done
  if (needed_fields.empty())
    return true;
  // There are no constraints for these fields so we get to do what we want
  instances.resize(instances.size() + 1);
  LayoutConstraintSet creation_constraints = our_constraints;
  std::vector<FieldID> creation_fields;
  DefaultMapper::default_policy_select_instance_fields(ctx, req, needed_fields,
                                                       creation_fields);
  creation_constraints.add_constraint(
      FieldConstraint(creation_fields, false /*contig*/, false /*inorder*/));
  if (!dsl_default_make_instance(ctx, target_memory, creation_constraints,
                                 instances.back(), TASK_MAPPING, force_new_instances,
                                 true /*meets*/, req, footprint))
    return false;
  return true;
}

bool NSMapper::dsl_default_make_instance(MapperContext ctx,
                                         Memory target_memory, const LayoutConstraintSet &constraints,
                                         PhysicalInstance &result, MappingKind kind, bool force_new, bool meets,
                                         const RegionRequirement &req, size_t *footprint)
//--------------------------------------------------------------------------
{
  bool created = true;
  Legion::LogicalRegion target_region =
      DefaultMapper::default_policy_select_instance_region(ctx, target_memory, req,
                                                           constraints, force_new, meets);
  bool tight_region_bounds = constraints.specialized_constraint.is_exact() || ((req.tag & DefaultMapper::EXACT_REGION) != 0);

  // TODO: deal with task layout constraints that require multiple
  // region requirements to be mapped to the same instance
  std::vector<Legion::LogicalRegion> target_regions(1, target_region);
  if (force_new ||
      ((req.privilege == LEGION_REDUCE) && (kind != COPY_MAPPING)))
  {
    if (!runtime->create_physical_instance(ctx, target_memory,
                                           constraints, target_regions, result, true /*acquire*/,
                                           0 /*priority*/, tight_region_bounds, footprint))
      return false;
  }
  else
  {
    if (!runtime->find_or_create_physical_instance(ctx,
                                                   target_memory, constraints, target_regions, result, created,
                                                   true /*acquire*/, 0 /*priority*/, tight_region_bounds, footprint))
      return false;
  }
  if (created)
  {
    int priority = DefaultMapper::default_policy_select_garbage_collection_priority(ctx,
                                                                                    kind, target_memory, result, meets, (req.privilege == LEGION_REDUCE));
    if ((priority != 0) && !result.is_external_instance())
      runtime->set_garbage_collection_priority(ctx, result, priority);
  }
  return true;
}

LayoutConstraintID NSMapper::dsl_default_policy_select_layout_constraints(MapperContext ctx,
                                                                          std::string task_name,
                                                                          unsigned idx,
                                                                          Memory target_memory,
                                                                          const RegionRequirement &req,
                                                                          MappingKind mapping_kind,
                                                                          bool needs_field_constraint_check,
                                                                          bool &force_new_instances)
//--------------------------------------------------------------------------
{
  // Do something special for reductions and
  // it is not an explicit region-to-region copy
  if ((req.privilege == LEGION_REDUCE) && (mapping_kind != COPY_MAPPING))
  {
    // Always make new reduction instances
    force_new_instances = true;
    std::pair<Memory::Kind, ReductionOpID> constraint_key(
        target_memory.kind(), req.redop);
    std::map<std::pair<Memory::Kind, ReductionOpID>, LayoutConstraintID>::
        const_iterator finder = reduction_constraint_cache.find(
            constraint_key);
    // No need to worry about field constraint checks here
    // since we don't actually have any field constraints
    if (finder != reduction_constraint_cache.end())
      return finder->second;
    LayoutConstraintSet constraints;
    dsl_default_policy_select_constraints(ctx, task_name, idx, constraints, target_memory, req);
    LayoutConstraintID result =
        runtime->register_layout(ctx, constraints);
    // Save the result
    reduction_constraint_cache[constraint_key] = result;
    return result;
  }
  // We always set force_new_instances to false since we are
  // deciding to optimize for minimizing memory usage instead
  // of avoiding Write-After-Read (WAR) dependences
  force_new_instances = false;
  // See if we've already made a constraint set for this layout
  std::pair<Memory::Kind, FieldSpace> constraint_key(target_memory.kind(),
                                                     req.region.get_field_space());
  std::map<std::pair<Memory::Kind, FieldSpace>, LayoutConstraintID>::
      const_iterator finder = layout_constraint_cache.find(constraint_key);
  if (finder != layout_constraint_cache.end())
  {
    // If we don't need a constraint check we are already good
    if (!needs_field_constraint_check)
      return finder->second;
    // Check that the fields still are the same, if not, fall through
    // so that we make a new set of constraints
    const LayoutConstraintSet &old_constraints =
        runtime->find_layout_constraints(ctx, finder->second);
    // Should be only one unless things have changed
    const std::vector<FieldID> &old_set =
        old_constraints.field_constraint.get_field_set();
    // Check to make sure the field sets are still the same
    std::vector<FieldID> new_fields;
    runtime->get_field_space_fields(ctx,
                                    constraint_key.second, new_fields);
    if (new_fields.size() == old_set.size())
    {
      std::set<FieldID> old_fields(old_set.begin(), old_set.end());
      bool still_equal = true;
      for (unsigned idx = 0; idx < new_fields.size(); idx++)
      {
        if (old_fields.find(new_fields[idx]) == old_fields.end())
        {
          still_equal = false;
          break;
        }
      }
      if (still_equal)
        return finder->second;
    }
    // Otherwise we fall through and make a new constraint which
    // will also update the cache
  }
  // Fill in the constraints
  LayoutConstraintSet constraints;
  dsl_default_policy_select_constraints(ctx, task_name, idx, constraints, target_memory, req);
  // Do the registration
  LayoutConstraintID result =
      runtime->register_layout(ctx, constraints);
  // Record our results, there is a benign race here as another mapper
  // call could have registered the exact same registration constraints
  // here if we were preempted during the registration call. The
  // constraint sets are identical though so it's all good.
  layout_constraint_cache[constraint_key] = result;
  return result;
}

void NSMapper::dsl_default_policy_select_constraints(MapperContext ctx,
                                                     std::string task_name, unsigned idx,
                                                     LayoutConstraintSet &constraints, Memory target_memory,
                                                     const RegionRequirement &req)
//--------------------------------------------------------------------------
{
  Memory::Kind target_memory_kind = target_memory.kind();

  ConstraintsNode dsl_constraint;
  ConstraintsNode *dsl_constraint_pt;
  if (use_semantic_name)
  {
    std::vector<std::string> path;
    get_handle_names(ctx, req, path);
    dsl_constraint_pt = tree_result.query_constraint(task_name, path, target_memory_kind);
  }
  else
  {
    dsl_constraint_pt = tree_result.query_constraint(task_name, {std::to_string(idx)}, target_memory_kind);
  }
  if (dsl_constraint_pt != NULL)
  {
    dsl_constraint = *dsl_constraint_pt;
    // log_mapper.debug() << "dsl_constraint specified by the user";

    Legion::IndexSpace is = req.region.get_index_space();
    Legion::Domain domain = runtime->get_index_space_domain(ctx, is);
    int dim = domain.get_dim();
    std::vector<Legion::DimensionKind> dimension_ordering(dim + 1);

    if (dsl_constraint.reverse)
    {
      // log_mapper.debug() << "dsl_constraint.reverse = true";
      if (dsl_constraint.aos)
      {
        // log_mapper.debug() << "dsl_constraint.aos = true";
        for (auto i = 0; i < dim; ++i)
        {
          dimension_ordering[dim - i] =
              static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
        }
        dimension_ordering[0] = LEGION_DIM_F;
      }
      else
      {
        // log_mapper.debug() << "dsl_constraint.aos = false";
        for (auto i = 0; i < dim; ++i)
        {
          dimension_ordering[dim - i - 1] =
              static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
        }
        dimension_ordering[dim] = LEGION_DIM_F; // soa
      }
    }
    else
    {
      // log_mapper.debug() << "dsl_constraint.reverse = false";
      if (dsl_constraint.aos)
      {
        // log_mapper.debug() << "dsl_constraint.aos = true";
        for (int i = 1; i < dim + 1; ++i)
        {
          dimension_ordering[i] =
              static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
        }
        dimension_ordering[0] = LEGION_DIM_F; // aos
      }
      else
      {
        // log_mapper.debug() << "dsl_constraint.aos = false";
        // DefaultMapper's choice
        for (auto i = 0; i < dim; ++i)
        {
          dimension_ordering[i] =
              static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
        }
        dimension_ordering[dim] = LEGION_DIM_F; // soa
      }
    }
    constraints.add_constraint(Legion::OrderingConstraint(dimension_ordering, false /*contiguous*/));
    // If we were requested to have an alignment, add the constraint.
    if (dsl_constraint.align)
    {
      // log_mapper.debug() << "dsl_constraint.align = true";
      for (auto it : req.privilege_fields)
      {
        constraints.add_constraint(Legion::AlignmentConstraint(it,
                                                               myop2legion(dsl_constraint.align_op), dsl_constraint.align_int));
      }
    }

    // Exact Region Constraint
    bool special_exact = dsl_constraint.exact;
    if (dsl_constraint.compact)
    {
      // sparse instance; we use SpecializedConstraint, which unfortunately has to override the default mapper
      // log_mapper.debug() << "dsl_constraint.compact = true";
      assert(req.privilege != LEGION_REDUCE);
      constraints.add_constraint(SpecializedConstraint(LEGION_COMPACT_SPECIALIZE, 0, false,
                                                       special_exact));
    }
  }
  DefaultMapper::default_policy_select_constraints(ctx, constraints, target_memory, req);
}

void NSMapper::map_replicate_task(const MapperContext ctx,
                                  const Task &task,
                                  const MapTaskInput &input,
                                  const MapTaskOutput &def_output,
                                  MapReplicateTaskOutput &output)
//--------------------------------------------------------------------------
{
  // Differs from DefaultMapper:
  // 1) remove assertion for replicating only top-level task
  // 2) default_find_preferred_variant not requiring tight bound (allowing finding multiple variants)
  // 3) support node=1, comment out the DEBUG_CTRL_REPL
  // 4) assume the variant is always replicable
  assert(task.regions.size() == 0);
  const Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  const VariantInfo chosen = default_find_preferred_variant(task, ctx,
                                                            false /*no need for tight bound*/,
                                                            true /*cache*/, target_kind);
  // if (chosen.is_replicable)
  if (true)
  {
    const std::vector<Processor> &remote_procs =
        remote_procs_by_kind(target_kind);
    // Place on replicate on each node by default
    assert(remote_procs.size() == total_nodes);
    output.task_mappings.resize(total_nodes, def_output);
    // Only check for MPI interop case when dealing with CPUs
    if ((target_kind == Processor::LOC_PROC) &&
        runtime->is_MPI_interop_configured(ctx))
    {
      // Check to see if we're interoperating with MPI
      const std::map<AddressSpace, int /*rank*/> &mpi_interop_mapping =
          runtime->find_reverse_MPI_mapping(ctx);
      // If we're interoperating with MPI make the shards align with ranks
      assert(mpi_interop_mapping.size() == total_nodes);
      for (std::vector<Processor>::const_iterator it =
               remote_procs.begin();
           it != remote_procs.end(); it++)
      {
        AddressSpace space = it->address_space();
        std::map<AddressSpace, int>::const_iterator finder =
            mpi_interop_mapping.find(space);
        assert(finder != mpi_interop_mapping.end());
        assert(finder->second < int(output.task_mappings.size()));
        output.task_mappings[finder->second].target_procs.push_back(*it);
      }
    }
    else
    {
      // Otherwise we can just assign shards based on address space
      if (total_nodes > 1)
      {
        for (std::vector<Processor>::const_iterator it =
                 remote_procs.begin();
             it != remote_procs.end(); it++)
        {
          AddressSpace space = it->address_space();
          assert(space < output.task_mappings.size());
          output.task_mappings[space].target_procs.push_back(*it);
        }
      }
      // #ifdef DEBUG_CTRL_REPL
      else
      {
        const std::vector<Processor> &local_procs =
            local_procs_by_kind(target_kind);
        output.task_mappings.resize(local_cpus.size());
        unsigned index = 0;
        for (std::vector<Processor>::const_iterator it =
                 local_procs.begin();
             it != local_procs.end(); it++, index++)
          output.task_mappings[index].target_procs.push_back(*it);
      }
      // #endif
    }
    // Indicate that we want to do control replication by filling
    // in the control replication map with our chosen processors
    // Also set our chosen variant
    if (total_nodes > 1)
    {
      output.control_replication_map.resize(total_nodes);
      for (unsigned idx = 0; idx < total_nodes; idx++)
      {
        output.task_mappings[idx].chosen_variant = chosen.variant;
        output.control_replication_map[idx] =
            output.task_mappings[idx].target_procs[0];
      }
    }
    // #ifdef DEBUG_CTRL_REPL
    else
    {
      const std::vector<Processor> &local_procs =
          local_procs_by_kind(target_kind);
      output.control_replication_map.resize(local_procs.size());
      for (unsigned idx = 0; idx < local_procs.size(); idx++)
      {
        output.task_mappings[idx].chosen_variant = chosen.variant;
        output.control_replication_map[idx] =
            output.task_mappings[idx].target_procs[0];
      }
    }
    // #endif
  }
  else
  {
    // log_mapper.warning("WARNING: Default mapper was unable to locate "
    //                    "a replicable task variant for the top-level "
    //                    "task during a multi-node execution! We STRONGLY "
    //                    "encourage users to make their top-level tasks "
    //                    "replicable to avoid sequential bottlenecks on "
    //                    "one node during the execution of an application!");
    output.task_mappings.resize(1);
    map_task(ctx, task, input, output.task_mappings[0]);
  }
}

void NSMapper::report_profiling(const MapperContext ctx,
                                const Task &task,
                                const TaskProfilingInfo &input)
{
  // We should only get profiling responses if we've enabled backpressuring.
  std::string task_name = task.get_task_name();
  assert(NSMapper::backpressure && tree_result.query_max_instance(task_name) > 0);
  bool is_index_launch = task.is_index_space && task.get_slice_domain().get_volume() > 1;
  auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
  // All our tasks should complete successfully.
  assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
  // Clean up after ourselves.
  delete prof;
  // Backpressured tasks are launched in a loop, and are kept on the originating processor.
  // So, we'll use orig_proc to index into the queue.
  auto &inflight = this->backPressureQueue[task.orig_proc];
  MapperEvent event;
  // Find this task in the queue.
  for (auto it = inflight.begin(); it != inflight.end(); it++)
  {
    if (is_index_launch)
    {
      if (it->id == std::make_pair(task.get_slice_domain(), task.get_context_index()))
      {
        event = it->event;
        inflight.erase(it);
        break;
      }
    }
    else
    {
      if (it->id2 == task.get_unique_id())
      {
        event = it->event;
        inflight.erase(it);
        break;
      }
    }
  }
  // Assert that we found a valid event.
  assert(event.exists());
  // Finally, trigger the event for anyone waiting on it.
  this->runtime->trigger_mapper_event(ctx, event);
}

// In select_tasks_to_map, we attempt to perform backpressuring on tasks that
// need to be backpressured.

void NSMapper::select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput &input,
                                   SelectMappingOutput &output)
{
  if (NSMapper::backpressure == false)
  {
    DefaultMapper::select_tasks_to_map(ctx, input, output);
  }
  else
  {
    // Mark when we are potentially scheduling tasks.
    auto schedTime = std::chrono::high_resolution_clock::now();
    // Create an event that we will return in case we schedule nothing.
    MapperEvent returnEvent;
    // Also maintain a time point of the best return event. We want this function
    // to get invoked as soon as any backpressure task finishes, so we'll use the
    // completion event for the earliest one.
    auto returnTime = std::chrono::high_resolution_clock::time_point::max();

    // Find the depth of the deepest task.
    int max_depth = 0;
    for (std::list<const Task *>::const_iterator it =
             input.ready_tasks.begin();
         it != input.ready_tasks.end(); it++)
    {
      int depth = (*it)->get_depth();
      if (depth > max_depth)
        max_depth = depth;
    }
    unsigned count = 0;
    // Only schedule tasks from the max depth in any pass.
    for (std::list<const Task *>::const_iterator it =
             input.ready_tasks.begin();
         (count < max_schedule_count) &&
         (it != input.ready_tasks.end());
         it++)
    {
      auto task = *it;
      bool schedule = true;
      std::string task_name = task->get_task_name();
      bool is_index_launch = task->is_index_space && task->get_slice_domain().get_volume() > 1;
      int max_num = tree_result.query_max_instance(task_name);
      if (max_num > 0)
      {
        // See how many tasks we have in flight. Again, we use the orig_proc here
        // rather than target_proc to match with our heuristics for where serial task
        // launch loops go.
        std::deque<InFlightTask> inflight = this->backPressureQueue[task->orig_proc];
        if ((int)inflight.size() == max_num)
        {
          // We've hit the cap, so we can't schedule any more tasks.
          schedule = false;
          // As a heuristic, we'll wait on the first mapper event to
          // finish, as it's likely that one will finish first. We'll also
          // try to get a task that will complete before the current best.
          auto front = inflight.front();
          if (front.schedTime < returnTime)
          {
            returnEvent = front.event;
            returnTime = front.schedTime;
          }
        }
        else
        {
          // Otherwise, we can schedule the task. Create a new event
          // and queue it up on the processor.
          if (is_index_launch)
          {
            InFlightTask a;
            a.id = std::make_pair(task->get_slice_domain(), task->get_context_index());
            // a.id2 = task->get_unique_id();
            a.event = this->runtime->create_mapper_event(ctx);
            a.schedTime = schedTime;
            this->backPressureQueue[task->orig_proc].push_back(a);
          }
          else
          {
            InFlightTask a;
            // a.id = Domain::NO_DOMAIN,
            a.id2 = task->get_unique_id();
            a.event = this->runtime->create_mapper_event(ctx);
            a.schedTime = schedTime;
            this->backPressureQueue[task->orig_proc].push_back(a);
          }
        }
      }
      // Schedule tasks that are valid and have the target depth.
      if (schedule && (*it)->get_depth() == max_depth)
      {
        output.map_tasks.insert(*it);
        count++;
      }
    }
    // If we didn't schedule any tasks, tell the runtime to ask us again when
    // our return event triggers.
    if (output.map_tasks.empty())
    {
      assert(returnEvent.exists());
      output.deferral_event = returnEvent;
    }
  }
}

Mapper::MapperSyncModel NSMapper::get_mapper_sync_model() const
{
  // If we're going to attempt to backpressure tasks, then we need to use
  // a sync model with high gaurantees.
  if (NSMapper::backpressure == true)
  {
    return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
  }
  // Otherwise, we can do whatever the default mapper is doing.
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void NSMapper::select_sharding_functor(
    const MapperContext ctx,
    const Task &task,
    const SelectShardingFunctorInput &input,
    SelectShardingFunctorOutput &output)
{
  std::string task_name = task.get_task_name();
  if (task2sid.count(task_name) > 0)
  {
    output.chosen_functor = task2sid.at(task_name);
    return;
  }
  Processor::Kind proc_kind = task.current_proc.kind();
  std::string proc_kind_string = processor_kind_to_string(proc_kind);
  if (task2sid.count(proc_kind_string) > 0)
  {
    output.chosen_functor = task2sid.at(proc_kind_string);
    return;
  }
  assert(tree_result.should_fall_back(task_name, task.is_index_space, proc_kind) == true);
  // log_mapper.debug("No sharding functor found in select_sharding_functor %s, fall back to default", task.get_task_name());
  output.chosen_functor = 0; // default functor
}

void NSMapper::default_policy_select_sources(MapperContext ctx,
                                             const PhysicalInstance &target,
                                             const std::vector<PhysicalInstance> &sources,
                                             std::deque<PhysicalInstance> &ranking)
{
  // Let the default mapper sort the sources by bandwidth
  DefaultMapper::default_policy_select_sources(ctx, target, sources, ranking);

  if (this->select_source_by_bandwidth)
  {
    return;
  }

  // Give priority to those with better overlapping
  std::vector<std::pair<PhysicalInstance, unsigned /*size of intersection*/>>
      cover_ranking(sources.size());

  Legion::Domain target_domain = target.get_instance_domain();
  for (std::deque<PhysicalInstance>::const_reverse_iterator it = ranking.rbegin();
       it != ranking.rend(); it++)
  {
    const unsigned idx = it - ranking.rbegin();
    const PhysicalInstance &source = (*it);
    Legion::Domain source_domain = source.get_instance_domain();
    Legion::Domain intersection = source_domain.intersection(target_domain);
    cover_ranking[idx] = std::pair<PhysicalInstance, unsigned>(source, intersection.get_volume());
  }

  // Sort them by the size of intersecting area
  std::stable_sort(cover_ranking.begin(), cover_ranking.end(), physical_sort_func);

  // Iterate from largest intersection, bandwidth to smallest
  ranking.clear();
  for (std::vector<std::pair<PhysicalInstance, unsigned>>::
           const_reverse_iterator it = cover_ranking.rbegin();
       it != cover_ranking.rend(); it++)
  {
    ranking.push_back(it->first);
  }
}

void NSMapper::select_task_options(const MapperContext ctx,
                                   const Task &task,
                                   TaskOptions &output)
//--------------------------------------------------------------------------
{
  // log_mapper.debug("NSMapper select_task_options in %s", get_mapper_name());
  output.initial_proc = dsl_default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  output.stealable = stealing_enabled;
  // This is the best choice for the default mapper assuming
  // there is locality in the remote mapped tasks
  output.map_locally = map_locally;
  std::string task_name = task.get_task_name();
  if (tree_result.control_replicate.size() > 0) // user specifies how to control-replicate tasks
  {
    if (tree_result.control_replicate.count(task_name) > 0)
    {
      output.replicate = true;
    }
    else
    {
      output.replicate = false;
    }
    return;
  }
  // If user does not specify, then fall back to default heuristics
  // Control replicate the top-level task in multi-node settings
  // otherwise we do no control replication
#ifdef DEBUG_CTRL_REPL
  if (task.get_depth() == 0)
#else
  if ((total_nodes > 1) && (task.get_depth() == 0))
#endif
    output.replicate = replication_enabled;
  // output.replicate = false; // no replication for now..
  else
    output.replicate = false;
}

template <int DIM>
void NSMapper::dsl_decompose_points(std::vector<int> &index_launch_space,
                                    const DomainT<DIM, coord_t> &point_space,
                                    const std::vector<Processor> &targets_local,
                                    const std::vector<std::vector<Processor>> &targets_all,
                                    bool recurse, bool stealable,
                                    std::vector<TaskSlice> &slices,
                                    std::string task_name,
                                    bool control_replicated)
//--------------------------------------------------------------------------
{
  // log_mapper.debug() << "dsl_decompose_points, dim=" << DIM
  // << " point_space.volume()=" << point_space.volume()
  // << " point_space=[" << point_space.bounds.lo[0] << "," << point_space.bounds.hi[0] << "]";
  slices.reserve(point_space.volume());

  for (Realm::IndexSpaceIterator<DIM, coord_t> it(point_space); it.valid; it.step())
  {
    for (Legion::PointInRectIterator<DIM, coord_t> itr(it.rect); itr(); itr++)
    {
      const Legion::Point<DIM, coord_t> point = *itr;
      std::vector<int> index_point;
      // log_mapper.debug("slice point: ");
      for (int i = 0; i < DIM; i++)
      {
        index_point.push_back(point[i]);
        // log_mapper.debug() << point[i] << " ,";
      }
      if (control_replicated)
      {
        // printf("Control replicated\n");
        size_t slice_res =
            (size_t)tree_result.runindex(task_name, index_point, index_launch_space, targets_local[0].kind())[0][1];
        // log_mapper.debug("--> %ld", slice_res);
        assert(slice_res < targets_local.size());
        // Construct the output slice for Legion.
        Legion::DomainT<DIM, Legion::coord_t> slice;
        slice.bounds.lo = point;
        slice.bounds.hi = point;
        slice.sparsity = point_space.sparsity;
        if (!slice.dense())
        {
          slice = slice.tighten();
        }
        if (slice.volume() > 0)
        {
          TaskSlice ts;
          ts.domain = slice;
          ts.proc = targets_local[slice_res];
          ts.recurse = recurse;
          ts.stealable = stealable;
          slices.push_back(ts);
        }
      }
      else
      {
        // printf("Not control-replicated");
        std::vector<int> node_proc = tree_result.runindex(task_name, index_point, index_launch_space, targets_local[0].kind())[0];
        int node_id = node_proc[0];
        int proc_id = node_proc[1];
        assert(node_id < (int) targets_all.size());
        assert(proc_id < (int) targets_all[node_id].size());
        // Construct the output slice for Legion.
        Legion::DomainT<DIM, Legion::coord_t> slice;
        slice.bounds.lo = point;
        slice.bounds.hi = point;
        slice.sparsity = point_space.sparsity;
        if (!slice.dense())
        {
          slice = slice.tighten();
        }
        if (slice.volume() > 0)
        {
          TaskSlice ts;
          ts.domain = slice;
          ts.proc = targets_all[node_id][proc_id];
          ts.recurse = recurse;
          ts.stealable = stealable;
          slices.push_back(ts);
        }
      }
    }
  }
}

void NSMapper::dsl_slice_task(const Task &task,
                              const std::vector<Processor> &local,
                              const std::vector<std::vector<Processor>> &all,
                              const SliceTaskInput &input,
                              SliceTaskOutput &output)
//--------------------------------------------------------------------------
{
  std::string task_name = task.get_task_name();
  std::vector<int> launch_space;
  Legion::Domain task_index_domain = task.index_domain;
  bool control_replicated = task.get_parent_task()->get_total_shards() > 1;
  switch (task_index_domain.get_dim())
  {
#define DIMFUNC(DIM)                                                 \
  case DIM:                                                          \
  {                                                                  \
    const DomainT<DIM, coord_t> is = task_index_domain;              \
    for (int i = 0; i < DIM; i++)                                    \
    {                                                                \
      launch_space.push_back(is.bounds.hi[i] - is.bounds.lo[i] + 1); \
    }                                                                \
    break;                                                           \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  default:
    assert(false);
  }

  switch (input.domain.get_dim())
  {
#define BLOCK(DIM)                                                                \
  case DIM:                                                                       \
  {                                                                               \
    DomainT<DIM, coord_t> partial_point_space = input.domain;                     \
    dsl_decompose_points<DIM>(launch_space, partial_point_space, local, all,      \
                              false /*recurse*/, stealing_enabled, output.slices, \
                              task_name, control_replicated);                     \
    break;                                                                        \
  }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
  default: // don't support other dimensions right now
    assert(false);
  }
}

void NSMapper::slice_task(const MapperContext ctx,
                          const Task &task,
                          const SliceTaskInput &input,
                          SliceTaskOutput &output)
{
  // Whatever kind of processor we are is the one this task should
  // be scheduled on as determined by select initial task
  Processor::Kind target_kind =
      task.must_epoch_task ? local_proc.kind() : task.target_proc.kind();
  // log_mapper.debug("%d,%d:%d", target_kind, local_proc.kind(), task.target_proc.kind());
  if (tree_result.should_fall_back(std::string(task.get_task_name()), task.is_index_space, target_kind))
  {
    // log_mapper.debug("Use default slice_task for %s", task.get_task_name());
    DefaultMapper::slice_task(ctx, task, input, output);
    return;
  }
  switch (target_kind)
  {
  case Processor::LOC_PROC:
  {
    // log_mapper.debug("%d: CPU here", target_kind);
    dsl_slice_task(task, local_cpus, all_cpus, input, output);
    break;
  }
  case Processor::TOC_PROC:
  {
    // log_mapper.debug("%d: GPU here", target_kind);
    dsl_slice_task(task, local_gpus, all_gpus, input, output);
    break;
  }
  case Processor::IO_PROC:
  {
    // log_mapper.debug("%d: IO here", target_kind);
    dsl_slice_task(task, local_ios, all_ios, input, output);
    break;
  }
  case Processor::PY_PROC:
  {
    // log_mapper.debug("%d: PY here", target_kind);
    dsl_slice_task(task, local_pys, all_pys, input, output);
    break;
  }
  case Processor::PROC_SET:
  {
    // log_mapper.debug("%d: PROC here", target_kind);
    dsl_slice_task(task, local_procsets, all_procsets, input, output);
    break;
  }
  case Processor::OMP_PROC:
  {
    // log_mapper.debug("%d: OMP here", target_kind);
    dsl_slice_task(task, local_omps, all_omps, input, output);
    break;
  }
  default:
    assert(false); // unimplemented processor kind
  }
}

NSMapper::NSMapper(MapperRuntime *rt, Machine machine, Processor local, const char *mapper_name, bool first)
    : DefaultMapper(rt, machine, local, mapper_name)
{
  if (first)
  {
    std::string policy_file = get_policy_file();
    parse_policy_file(policy_file);
  }
  for (size_t i = 0; i < this->local_gpus.size(); i++)
  {
    query_best_memory_for_proc(this->local_gpus[i], Memory::GPU_FB_MEM);
    query_best_memory_for_proc(this->local_gpus[i], Memory::Z_COPY_MEM);
  }
  for (size_t i = 0; i < this->local_cpus.size(); i++)
  {
    query_best_memory_for_proc(this->local_cpus[i], Memory::SYSTEM_MEM);
    query_best_memory_for_proc(this->local_cpus[i], Memory::Z_COPY_MEM);
    query_best_memory_for_proc(this->local_cpus[i], Memory::SOCKET_MEM);
    query_best_memory_for_proc(this->local_cpus[i], Memory::REGDMA_MEM);
  }
  for (size_t i = 0; i < this->local_omps.size(); i++)
  {
    query_best_memory_for_proc(this->local_omps[i], Memory::SYSTEM_MEM);
    query_best_memory_for_proc(this->local_omps[i], Memory::Z_COPY_MEM);
    query_best_memory_for_proc(this->local_omps[i], Memory::SOCKET_MEM);
    query_best_memory_for_proc(this->local_omps[i], Memory::REGDMA_MEM);
  }
  all_gpus.resize(total_nodes, std::vector<Processor>{});
  all_cpus.resize(total_nodes, std::vector<Processor>{});
  all_ios.resize(total_nodes, std::vector<Processor>{});
  all_procsets.resize(total_nodes, std::vector<Processor>{});
  all_omps.resize(total_nodes, std::vector<Processor>{});
  all_pys.resize(total_nodes, std::vector<Processor>{});
  Machine::ProcessorQuery all_procs(machine);
  for (Machine::ProcessorQuery::iterator it = all_procs.begin(); it != all_procs.end(); it++)
  {
    AddressSpace node = it->address_space();
    switch (it->kind())
    {
    case Processor::TOC_PROC:
    {
      all_gpus[node].push_back(*it);
      break;
    }
    case Processor::LOC_PROC:
    {
      all_cpus[node].push_back(*it);
      break;
    }
    case Processor::IO_PROC:
    {
      all_ios[node].push_back(*it);
      break;
    }
    case Processor::PY_PROC:
    {
      all_pys[node].push_back(*it);
      break;
    }
    case Processor::PROC_SET:
    {
      all_procsets[node].push_back(*it);
      break;
    }
    case Processor::OMP_PROC:
    {
      all_omps[node].push_back(*it);
      break;
    }
    default: // ignore anything else
      break;
    }
  }
}

void NSMapper::create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  // log_mapper.debug("Inside create_mappers local_procs.size() = %ld", local_procs.size());
  bool use_logging_wrapper = false;
  auto args = Runtime::get_input_args();
  NSMapper::backpressure = true;
  NSMapper::use_semantic_name = false;
  NSMapper::untrackValidRegions = true;
  NSMapper::select_source_by_bandwidth = false;
  for (auto idx = 0; idx < args.argc; ++idx)
  {
    if (strcmp(args.argv[idx], "-wrapper") == 0)
    {
      use_logging_wrapper = true;
    }
    // todo: in the final version, change tm to be the formal name of DSLMapper
    if (strcmp(args.argv[idx], "-tm:disable_backpressure") == 0)
    {
      NSMapper::backpressure = false;
    }
    if (strcmp(args.argv[idx], "-tm:disable_untrack_valid_regions") == 0)
    {
      NSMapper::untrackValidRegions = false;
    }
    if (strcmp(args.argv[idx], "-tm:use_semantic_name") == 0)
    {
      NSMapper::use_semantic_name = true;
    }
    if (strcmp(args.argv[idx], "-tm:select_source_by_bandwidth") == 0)
    {
      NSMapper::select_source_by_bandwidth = true;
    }
  }
#ifdef DEBUG_COMMAND_LINE
  printf("use_logging_wrapper = %s\n", use_logging_wrapper ? "true" : "false");
  printf("backpressure = %s\n", NSMapper::backpressure ? "true" : "false");
  printf("untrackValidRegions = %s\n", NSMapper::untrackValidRegions ? "true" : "false");
  printf("use_semantic_name = %s\n", NSMapper::use_semantic_name ? "true" : "false");
  printf("select_source_by_bandwidth = %s\n", NSMapper::select_source_by_bandwidth ? "true" : "false");
#endif
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++)
  {
    NSMapper *mapper = NULL;
    if (it == local_procs.begin())
    {
      mapper = new NSMapper(runtime->get_mapper_runtime(), machine, *it, "ns_mapper", true);
      mapper->register_user_sharding_functors(runtime);
      mapper->build_proc_idx_cache();
    }
    else
    {
      mapper = new NSMapper(runtime->get_mapper_runtime(), machine, *it, "ns_mapper", false);
      mapper->build_proc_idx_cache();
    }
    if (use_logging_wrapper)
    {
      runtime->replace_default_mapper(new Mapping::LoggingWrapper(mapper), (NSMapper::backpressure ? (Processor::NO_PROC) : (*it)));
    }
    else
    {
      runtime->replace_default_mapper(mapper, (NSMapper::backpressure ? (Processor::NO_PROC) : (*it)));
    }
    if (NSMapper::backpressure)
    {
      break;
    }
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(NSMapper::create_mappers);
}

namespace Legion
{
  namespace Internal
  {
    /////////////////////////////////////////////////////////////
    // User Sharding Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UserShardingFunctor::UserShardingFunctor(std::string takename_, const Tree2Legion &tree_)
        : ShardingFunctor(), taskname(takename_), tree(tree_)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor::UserShardingFunctor(
        const UserShardingFunctor &rhs)
        : ShardingFunctor()
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor::~UserShardingFunctor(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor &UserShardingFunctor::operator=(
        const UserShardingFunctor &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    ShardID UserShardingFunctor::shard(const DomainPoint &point,
                                       const Legion::Domain &full_space,
                                       const size_t total_shards)
    //--------------------------------------------------------------------------
    {
      // printf("Sharded\n");
#ifdef DEBUG_LEGION
      assert(point.get_dim() == full_space.get_dim());
#endif
      // log_mapper.debug("shard dim: %d, total_shards: %ld", point.get_dim(), total_shards);
      // size_t node_num = Machine::get_machine().get_address_space_count();
      switch (point.get_dim())
      {
#define DIMFUNC(DIM)                                                 \
  case DIM:                                                          \
  {                                                                  \
    const DomainT<DIM, coord_t> is = full_space;                     \
    const Legion::Point<DIM, coord_t> p1 = point;                    \
    std::vector<int> index_point, launch_space;                      \
    for (int i = 0; i < DIM; i++)                                    \
    {                                                                \
      index_point.push_back(p1[i]);                                  \
      launch_space.push_back(is.bounds.hi[i] - is.bounds.lo[i] + 1); \
    }                                                                \
    return tree.runindex(taskname, index_point, launch_space)[0][0]; \
  }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
      }
      // log_mapper.debug("shard: should never reach");
      assert(false);
      return 0;
    }
  }
}

legion_equality_kind_t myop2legion(BinOpEnum myop)
{
  // BIGGER,	SMALLER,	GE,	LE,	EQ,	NEQ,:
  switch (myop)
  {
  case BIGGER:
    return LEGION_GT_EK;
  case SMALLER:
    return LEGION_LT_EK;
  case GE:
    return LEGION_GE_EK;
  case LE:
    return LEGION_LE_EK;
  case EQ:
    return LEGION_EQ_EK;
  case NEQ:
    return LEGION_NE_EK;
  default:
    break;
  }
  assert(false);
  return LEGION_EQ_EK;
}

#include "compiler/tree.cpp"
