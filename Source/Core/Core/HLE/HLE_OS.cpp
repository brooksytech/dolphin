// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HLE/HLE_OS.h"

#include <memory>
#include <string>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Common/MsgHandler.h"
#include "Common/StringUtil.h"
#include "Core/HLE/HLE_VarArgs.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"

namespace HLE_OS
{
enum class ParameterType : bool
{
  ParameterList = false,
  VariableArgumentList = true
};

std::string GetStringVA(Core::System& system, u32 str_reg = 3,
                        ParameterType parameter_type = ParameterType::ParameterList);
void HLE_GeneralDebugPrint(ParameterType parameter_type);
void HLE_LogDPrint(ParameterType parameter_type);
void HLE_LogFPrint(ParameterType parameter_type);

void HLE_OSPanic()
{
  auto& system = Core::System::GetInstance();
  auto& ppc_state = system.GetPPCState();

  std::string error = GetStringVA(system);
  std::string msg = GetStringVA(system, 5);

  StringPopBackIf(&error, '\n');
  StringPopBackIf(&msg, '\n');

  PanicAlertFmt("OSPanic: {}: {}", error, msg);
  ERROR_LOG_FMT(OSREPORT_HLE, "{:08x}->{:08x}| OSPanic: {}: {}", LR(ppc_state), ppc_state.pc, error,
                msg);

  ppc_state.npc = LR(ppc_state);
}

// Generalized function for printing formatted string.
void HLE_GeneralDebugPrint(ParameterType parameter_type)
{
  auto& system = Core::System::GetInstance();
  auto& ppc_state = system.GetPPCState();

  std::string report_message;

  // Is gpr3 pointing to a pointer (including nullptr) rather than an ASCII string
  if (PowerPC::HostIsRAMAddress(ppc_state.gpr[3]) &&
      (PowerPC::HostIsRAMAddress(PowerPC::HostRead_U32(ppc_state.gpr[3])) ||
       PowerPC::HostRead_U32(ppc_state.gpr[3]) == 0))
  {
    if (PowerPC::HostIsRAMAddress(ppc_state.gpr[4]))
    {
      // ___blank(void* this, const char* fmt, ...);
      report_message = GetStringVA(system, 4, parameter_type);
    }
    else
    {
      // ___blank(void* this, int log_type, const char* fmt, ...);
      report_message = GetStringVA(system, 5, parameter_type);
    }
  }
  else
  {
    if (PowerPC::HostIsRAMAddress(ppc_state.gpr[3]))
    {
      // ___blank(const char* fmt, ...);
      report_message = GetStringVA(system, 3, parameter_type);
    }
    else
    {
      // ___blank(int log_type, const char* fmt, ...);
      report_message = GetStringVA(system, 4, parameter_type);
    }
  }

  StringPopBackIf(&report_message, '\n');

  NOTICE_LOG_FMT(OSREPORT_HLE, "{:08x}->{:08x}| {}", LR(ppc_state), ppc_state.pc,
                 SHIFTJISToUTF8(report_message));
}

// Generalized function for printing formatted string using parameter list.
void HLE_GeneralDebugPrint()
{
  HLE_GeneralDebugPrint(ParameterType::ParameterList);
}

// Generalized function for printing formatted string using va_list.
void HLE_GeneralDebugVPrint()
{
  HLE_GeneralDebugPrint(ParameterType::VariableArgumentList);
}

// __write_console(int fd, const void* buffer, const u32* size)
void HLE_write_console()
{
  auto& system = Core::System::GetInstance();
  auto& ppc_state = system.GetPPCState();

  std::string report_message = GetStringVA(system, 4);
  if (PowerPC::HostIsRAMAddress(ppc_state.gpr[5]))
  {
    const u32 size = PowerPC::Read_U32(ppc_state.gpr[5]);
    if (size > report_message.size())
      WARN_LOG_FMT(OSREPORT_HLE, "__write_console uses an invalid size of {:#010x}", size);
    else if (size == 0)
      WARN_LOG_FMT(OSREPORT_HLE, "__write_console uses a size of zero");
    else
      report_message = report_message.substr(0, size);
  }
  else
  {
    ERROR_LOG_FMT(OSREPORT_HLE, "__write_console uses an unreachable size pointer");
  }

  StringPopBackIf(&report_message, '\n');

  NOTICE_LOG_FMT(OSREPORT_HLE, "{:08x}->{:08x}| {}", LR(ppc_state), ppc_state.pc,
                 SHIFTJISToUTF8(report_message));
}

// Log (v)dprintf message if fd is 1 (stdout) or 2 (stderr)
void HLE_LogDPrint(ParameterType parameter_type)
{
  auto& system = Core::System::GetInstance();
  auto& ppc_state = system.GetPPCState();

  if (ppc_state.gpr[3] != 1 && ppc_state.gpr[3] != 2)
    return;

  std::string report_message = GetStringVA(system, 4, parameter_type);
  StringPopBackIf(&report_message, '\n');
  NOTICE_LOG_FMT(OSREPORT_HLE, "{:08x}->{:08x}| {}", LR(ppc_state), ppc_state.pc,
                 SHIFTJISToUTF8(report_message));
}

// Log dprintf message
//  -> int dprintf(int fd, const char* format, ...);
void HLE_LogDPrint()
{
  HLE_LogDPrint(ParameterType::ParameterList);
}

// Log vdprintf message
//  -> int vdprintf(int fd, const char* format, va_list ap);
void HLE_LogVDPrint()
{
  HLE_LogDPrint(ParameterType::VariableArgumentList);
}

// Log (v)fprintf message if FILE is stdout or stderr
void HLE_LogFPrint(ParameterType parameter_type)
{
  auto& system = Core::System::GetInstance();
  auto& ppc_state = system.GetPPCState();

  // The structure FILE is implementation defined.
  // Both libogc and Dolphin SDK seem to store the fd at the same address.
  int fd = -1;
  if (PowerPC::HostIsRAMAddress(ppc_state.gpr[3]) &&
      PowerPC::HostIsRAMAddress(ppc_state.gpr[3] + 0xF))
  {
    // The fd is stored as a short at FILE+0xE.
    fd = static_cast<short>(PowerPC::HostRead_U16(ppc_state.gpr[3] + 0xE));
  }
  if (fd != 1 && fd != 2)
  {
    // On RVL SDK it seems stored at FILE+0x2.
    fd = static_cast<short>(PowerPC::HostRead_U16(ppc_state.gpr[3] + 0x2));
  }
  if (fd != 1 && fd != 2)
    return;

  std::string report_message = GetStringVA(system, 4, parameter_type);
  StringPopBackIf(&report_message, '\n');
  NOTICE_LOG_FMT(OSREPORT_HLE, "{:08x}->{:08x}| {}", LR(ppc_state), ppc_state.pc,
                 SHIFTJISToUTF8(report_message));
}

// Log fprintf message
//  -> int fprintf(FILE* stream, const char* format, ...);
void HLE_LogFPrint()
{
  HLE_LogFPrint(ParameterType::ParameterList);
}

// Log vfprintf message
//  -> int vfprintf(FILE* stream, const char* format, va_list ap);
void HLE_LogVFPrint()
{
  HLE_LogFPrint(ParameterType::VariableArgumentList);
}

std::string GetStringVA(Core::System& system, u32 str_reg, ParameterType parameter_type)
{
  auto& ppc_state = system.GetPPCState();

  std::string ArgumentBuffer;
  std::string result;
  std::string string = PowerPC::HostGetString(ppc_state.gpr[str_reg]);
  auto ap =
      parameter_type == ParameterType::VariableArgumentList ?
          std::make_unique<HLE::SystemVABI::VAListStruct>(system, ppc_state.gpr[str_reg + 1]) :
          std::make_unique<HLE::SystemVABI::VAList>(system, ppc_state.gpr[1] + 0x8, str_reg + 1);

  for (size_t i = 0; i < string.size(); i++)
  {
    if (string[i] == '%')
    {
      ArgumentBuffer = '%';
      i++;
      if (string[i] == '%')
      {
        result += '%';
        continue;
      }

      while (i < string.size() &&
             (string[i] < 'A' || string[i] > 'z' || string[i] == 'l' || string[i] == '-'))
      {
        ArgumentBuffer += string[i++];
      }
      if (i >= string.size())
        break;

      ArgumentBuffer += string[i];

      switch (string[i])
      {
      case 's':
        result += StringFromFormat(ArgumentBuffer.c_str(),
                                   PowerPC::HostGetString(ap->GetArgT<u32>()).c_str());
        break;

      case 'a':
      case 'A':
      case 'e':
      case 'E':
      case 'f':
      case 'F':
      case 'g':
      case 'G':
        result += StringFromFormat(ArgumentBuffer.c_str(), ap->GetArgT<double>());
        break;

      case 'p':
        // Override, so 64bit Dolphin prints 32bit pointers, since the ppc is 32bit :)
        result += StringFromFormat("%x", ap->GetArgT<u32>());
        break;

      case 'n':
        // %n doesn't output anything, so the result variable is untouched
        // the actual PPC function will take care of the memory write
        break;

      default:
        if (string[i - 1] == 'l' && string[i - 2] == 'l')
          result += StringFromFormat(ArgumentBuffer.c_str(), ap->GetArgT<u64>());
        else
          result += StringFromFormat(ArgumentBuffer.c_str(), ap->GetArgT<u32>());
        break;
      }
    }
    else
    {
      result += string[i];
    }
  }

  return result;
}

}  // namespace HLE_OS
