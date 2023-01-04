// Copyright 2022 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VKScheduler.h"
#include <Common/Assert.h>
#include "Common/Thread.h"
#include "StateTracker.h"

namespace Vulkan
{
Scheduler::Scheduler()
    : m_commandBufferManager(std::make_unique<CommandBufferManager>()),
      m_submit_loop(std::make_unique<Common::BlockingLoop>())
{
  AcquireNewChunk();

  m_worker = std::thread([this]() {
    Common::SetCurrentThreadName("Vulkan CS Thread");
    WorkerThread();
  });
}

Scheduler::~Scheduler()
{
  m_submit_loop->Stop();
  m_worker.join();
}

bool Scheduler::Initialize()
{
  return m_commandBufferManager->Initialize();
}

void Scheduler::CommandChunk::ExecuteAll(CommandBufferManager* cmdbuf)
{
  auto command = first;
  while (command != nullptr)
  {
    auto next = command->GetNext();
    command->Execute(cmdbuf);
    command->~Command();
    command = next;
  }
  command_offset = 0;
  first = nullptr;
  last = nullptr;
}

void Scheduler::AcquireNewChunk()
{
  std::scoped_lock lock{m_reserve_mutex};
  if (m_chunk_reserve.empty())
  {
    m_chunk = std::make_unique<CommandChunk>();
    return;
  }
  m_chunk = std::move(m_chunk_reserve.back());
  m_chunk_reserve.pop_back();
}

void Scheduler::Flush()
{
  if (m_chunk->Empty())
    return;

  {
    std::scoped_lock lock{m_work_mutex};
    m_worker_idle = false;
    m_work_queue.push(std::move(m_chunk));
    m_submit_loop->Wakeup();
  }
  AcquireNewChunk();
}

void Scheduler::SyncWorker()
{
  Flush();
  std::unique_lock lock{m_work_mutex};
  m_idle_condvar.wait(lock, [this] { return m_worker_idle; });
}

void Scheduler::WorkerThread()
{
  m_submit_loop->Run([this]() {
    std::unique_ptr<CommandChunk> work;
    {
      std::scoped_lock lock{m_work_mutex};
      if (m_work_queue.empty())
      {
        m_worker_idle = true;
        m_idle_condvar.notify_all();
        m_submit_loop->AllowSleep();
        return;
      }
      work = std::move(m_work_queue.front());
      m_work_queue.pop();
    }

    work->ExecuteAll(m_commandBufferManager.get());
    {
      std::scoped_lock reserve_lock{m_reserve_mutex};
      m_chunk_reserve.push_back(std::move(work));
    }
    {
      std::scoped_lock lock{m_work_mutex};
      if (m_work_queue.empty())
      {
        m_worker_idle = true;
        m_idle_condvar.notify_all();
      }
    }
  });
}

void Scheduler::Shutdown()
{
  SyncWorker();
  SynchronizeSubmissionThread();
}

void Scheduler::SynchronizeSubmissionThread()
{
  SyncWorker();
  m_commandBufferManager->WaitForSubmitWorkerThreadIdle();
}

void Scheduler::WaitForFenceCounter(u64 counter)
{
  if (m_commandBufferManager->GetCompletedFenceCounter() >= counter)
    return;

  SyncWorker();
  m_commandBufferManager->WaitForFenceCounter(counter);
}

void Scheduler::SubmitCommandBuffer(bool submit_on_worker_thread, bool wait_for_completion,
                                    VkSwapchainKHR present_swap_chain, uint32_t present_image_index)
{
  const u64 fence_counter = ++m_current_fence_counter;
  Record([fence_counter, submit_on_worker_thread, wait_for_completion, present_swap_chain,
          present_image_index](CommandBufferManager* command_buffer_mgr) {
    command_buffer_mgr->GetStateTracker()->EndRenderPass();
    command_buffer_mgr->SubmitCommandBuffer(fence_counter, submit_on_worker_thread,
                                            wait_for_completion, present_swap_chain,
                                            present_image_index);
  });

  if (wait_for_completion) [[unlikely]]
    g_scheduler->WaitForFenceCounter(fence_counter);
  else
    Flush();
}

std::unique_ptr<Scheduler> g_scheduler;
}  // namespace Vulkan
