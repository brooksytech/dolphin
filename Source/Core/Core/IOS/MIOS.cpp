// Copyright 2017 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/IOS/MIOS.h"

#include <cstring>
#include <utility>

#include "Common/CommonTypes.h"
#include "Common/FileUtil.h"
#include "Common/Logging/Log.h"
#include "Common/MsgHandler.h"
#include "Common/Swap.h"
#include "Core/Config/MainSettings.h"
#include "Core/ConfigManager.h"
#include "Core/Core.h"
#include "Core/DSPEmulator.h"
#include "Core/HLE/HLE.h"
#include "Core/HW/DSP.h"
#include "Core/HW/DVD/DVDInterface.h"
#include "Core/HW/Memmap.h"
#include "Core/HW/SystemTimers.h"
#include "Core/HW/Wiimote.h"
#include "Core/Host.h"
#include "Core/PowerPC/PPCSymbolDB.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"

namespace IOS::HLE::MIOS
{
static void ReinitHardware()
{
  SConfig::GetInstance().bWii = false;

  // IOS clears mem2 and overwrites it with pseudo-random data (for security).
  auto& system = Core::System::GetInstance();
  auto& memory = system.GetMemory();
  std::memset(memory.GetEXRAM(), 0, memory.GetExRamSizeReal());
  // MIOS appears to only reset the DI and the PPC.
  // HACK However, resetting DI will reset the DTK config, which is set by the system menu
  // (and not by MIOS), causing games that use DTK to break.  Perhaps MIOS doesn't actually
  // reset DI fully, in such a way that the DTK config isn't cleared?
  // DVDInterface::ResetDrive(true);
  PowerPC::Reset();
  Wiimote::ResetAllWiimotes();
  // Note: this is specific to Dolphin and is required because we initialised it in Wii mode.
  DSP::Reinit(Config::Get(Config::MAIN_DSP_HLE));
  DSP::GetDSPEmulator()->Initialize(SConfig::GetInstance().bWii,
                                    Config::Get(Config::MAIN_DSP_THREAD));

  SystemTimers::ChangePPCClock(SystemTimers::Mode::GC);
}

constexpr u32 ADDRESS_INIT_SEMAPHORE = 0x30f8;

bool Load()
{
  auto& system = Core::System::GetInstance();
  auto& memory = system.GetMemory();
  memory.Write_U32(0x00000000, ADDRESS_INIT_SEMAPHORE);
  memory.Write_U32(0x09142001, 0x3180);

  ReinitHardware();
  NOTICE_LOG_FMT(IOS, "Reinitialised hardware.");

  // Load symbols for the IPL if they exist.
  if (!g_symbolDB.IsEmpty())
  {
    g_symbolDB.Clear();
    Host_NotifyMapLoaded();
  }
  if (g_symbolDB.LoadMap(File::GetUserPath(D_MAPS_IDX) + "mios-ipl.map"))
  {
    ::HLE::Clear();
    ::HLE::PatchFunctions(system);
    Host_NotifyMapLoaded();
  }

  auto& ppc_state = system.GetPPCState();
  const PowerPC::CoreMode core_mode = PowerPC::GetMode();
  PowerPC::SetMode(PowerPC::CoreMode::Interpreter);
  ppc_state.msr.Hex = 0;
  ppc_state.pc = 0x3400;
  NOTICE_LOG_FMT(IOS, "Loaded MIOS and bootstrapped PPC.");

  // IOS writes 0 to 0x30f8 before bootstrapping the PPC. Once started, the IPL eventually writes
  // 0xdeadbeef there, then waits for it to be cleared by IOS before continuing.
  while (memory.Read_U32(ADDRESS_INIT_SEMAPHORE) != 0xdeadbeef)
    PowerPC::SingleStep();
  PowerPC::SetMode(core_mode);

  memory.Write_U32(0x00000000, ADDRESS_INIT_SEMAPHORE);
  NOTICE_LOG_FMT(IOS, "IPL ready.");
  SConfig::GetInstance().m_is_mios = true;
  DVDInterface::UpdateRunningGameMetadata();
  SConfig::OnNewTitleLoad();
  return true;
}
}  // namespace IOS::HLE::MIOS
