// Copyright 2017 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "DolphinQt/Debugger/WatchWidget.h"

#include <QHeaderView>
#include <QInputDialog>
#include <QMenu>
#include <QTableWidget>
#include <QToolBar>
#include <QVBoxLayout>

#include "Common/FileUtil.h"
#include "Common/IniFile.h"
#include "Core/ConfigManager.h"
#include "Core/Core.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"

#include "DolphinQt/Host.h"
#include "DolphinQt/QtUtils/ModalMessageBox.h"
#include "DolphinQt/Resources.h"
#include "DolphinQt/Settings.h"

WatchWidget::WatchWidget(QWidget* parent) : QDockWidget(parent)
{
  // i18n: This kind of "watch" is used for watching emulated memory.
  // It's not related to timekeeping devices.
  setWindowTitle(tr("Watch"));
  setObjectName(QStringLiteral("watch"));

  setHidden(!Settings::Instance().IsWatchVisible() || !Settings::Instance().IsDebugModeEnabled());

  setAllowedAreas(Qt::AllDockWidgetAreas);

  CreateWidgets();

  auto& settings = Settings::GetQSettings();

  restoreGeometry(settings.value(QStringLiteral("watchwidget/geometry")).toByteArray());
  // macOS: setHidden() needs to be evaluated before setFloating() for proper window presentation
  // according to Settings
  setFloating(settings.value(QStringLiteral("watchwidget/floating")).toBool());

  ConnectWidgets();

  connect(&Settings::Instance(), &Settings::EmulationStateChanged, this, [this](Core::State state) {
    UpdateButtonsEnabled();
    if (state != Core::State::Starting)
      Update();
  });

  connect(Host::GetInstance(), &Host::UpdateDisasmDialog, this, &WatchWidget::Update);

  connect(&Settings::Instance(), &Settings::WatchVisibilityChanged, this,
          [this](bool visible) { setHidden(!visible); });

  connect(&Settings::Instance(), &Settings::DebugModeToggled, this,
          [this](bool enabled) { setHidden(!enabled || !Settings::Instance().IsWatchVisible()); });

  connect(&Settings::Instance(), &Settings::ThemeChanged, this, &WatchWidget::UpdateIcons);
  UpdateIcons();
}

WatchWidget::~WatchWidget()
{
  auto& settings = Settings::GetQSettings();

  settings.setValue(QStringLiteral("watchwidget/geometry"), saveGeometry());
  settings.setValue(QStringLiteral("watchwidget/floating"), isFloating());
}

void WatchWidget::CreateWidgets()
{
  m_toolbar = new QToolBar;
  m_toolbar->setContentsMargins(0, 0, 0, 0);
  m_toolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  m_table = new QTableWidget;
  m_table->setTabKeyNavigation(false);

  m_table->setContentsMargins(0, 0, 0, 0);
  m_table->setColumnCount(NUM_COLUMNS);
  m_table->verticalHeader()->setHidden(true);
  m_table->setContextMenuPolicy(Qt::CustomContextMenu);
  m_table->setSelectionMode(QAbstractItemView::ExtendedSelection);
  m_table->setSelectionBehavior(QAbstractItemView::SelectRows);

  m_new = m_toolbar->addAction(tr("New"), this, &WatchWidget::OnNewWatch);
  m_delete = m_toolbar->addAction(tr("Delete"), this, &WatchWidget::OnDelete);
  m_clear = m_toolbar->addAction(tr("Clear"), this, &WatchWidget::OnClear);
  m_load = m_toolbar->addAction(tr("Load"), this, &WatchWidget::OnLoad);
  m_save = m_toolbar->addAction(tr("Save"), this, &WatchWidget::OnSave);

  m_new->setEnabled(false);
  m_delete->setEnabled(false);
  m_clear->setEnabled(false);
  m_load->setEnabled(false);
  m_save->setEnabled(false);

  auto* layout = new QVBoxLayout;
  layout->setContentsMargins(2, 2, 2, 2);
  layout->setSpacing(0);
  layout->addWidget(m_toolbar);
  layout->addWidget(m_table);

  QWidget* widget = new QWidget;
  widget->setLayout(layout);

  setWidget(widget);
}

void WatchWidget::ConnectWidgets()
{
  connect(m_table, &QTableWidget::customContextMenuRequested, this, &WatchWidget::ShowContextMenu);
  connect(m_table, &QTableWidget::itemChanged, this, &WatchWidget::OnItemChanged);
}

void WatchWidget::UpdateIcons()
{
  // TODO: Create a "debugger_add_watch" icon
  m_new->setIcon(Resources::GetScaledThemeIcon("debugger_add_breakpoint"));
  m_delete->setIcon(Resources::GetScaledThemeIcon("debugger_delete"));
  m_clear->setIcon(Resources::GetScaledThemeIcon("debugger_clear"));
  m_load->setIcon(Resources::GetScaledThemeIcon("debugger_load"));
  m_save->setIcon(Resources::GetScaledThemeIcon("debugger_save"));
}

void WatchWidget::UpdateButtonsEnabled()
{
  if (!isVisible())
    return;

  const bool is_enabled = Core::IsRunning();
  m_new->setEnabled(is_enabled);
  m_delete->setEnabled(is_enabled);
  m_clear->setEnabled(is_enabled);
  m_load->setEnabled(is_enabled);
  m_save->setEnabled(is_enabled);
}

void WatchWidget::Update()
{
  if (!isVisible())
    return;

  m_updating = true;

  m_table->clear();

  int size = static_cast<int>(PowerPC::debug_interface.GetWatches().size());

  m_table->setRowCount(size + 1);

  m_table->setHorizontalHeaderLabels(
      {tr("Label"), tr("Address"), tr("Hexadecimal"),
       // i18n: The base 10 numeral system. Not related to non-integer numbers
       tr("Decimal"),
       // i18n: Data type used in computing
       tr("String"),
       // i18n: Floating-point (non-integer) number
       tr("Float"), tr("Locked")});

  for (int i = 0; i < size; i++)
  {
    const auto& entry = PowerPC::debug_interface.GetWatch(i);

    auto* label = new QTableWidgetItem(QString::fromStdString(entry.name));
    auto* address =
        new QTableWidgetItem(QStringLiteral("%1").arg(entry.address, 8, 16, QLatin1Char('0')));
    auto* hex = new QTableWidgetItem;
    auto* decimal = new QTableWidgetItem;
    auto* string = new QTableWidgetItem;
    auto* floatValue = new QTableWidgetItem;

    auto* lockValue = new QTableWidgetItem;
    lockValue->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);

    std::array<QTableWidgetItem*, NUM_COLUMNS> items = {label,  address,    hex,      decimal,
                                                        string, floatValue, lockValue};

    QBrush brush = QPalette().brush(QPalette::Text);

    if (!Core::IsRunning() || !PowerPC::HostIsRAMAddress(entry.address))
      brush.setColor(Qt::red);

    if (Core::IsRunning())
    {
      if (PowerPC::HostIsRAMAddress(entry.address))
      {
        hex->setText(QStringLiteral("%1").arg(PowerPC::HostRead_U32(entry.address), 8, 16,
                                              QLatin1Char('0')));
        decimal->setText(QString::number(PowerPC::HostRead_U32(entry.address)));
        string->setText(QString::fromStdString(PowerPC::HostGetString(entry.address, 32)));
        floatValue->setText(QString::number(PowerPC::HostRead_F32(entry.address)));
        lockValue->setCheckState(entry.locked ? Qt::Checked : Qt::Unchecked);
      }
    }

    address->setForeground(brush);
    string->setFlags(Qt::ItemIsEnabled);

    for (int column = 0; column < NUM_COLUMNS; column++)
    {
      auto* item = items[column];
      item->setData(Qt::UserRole, i);
      item->setData(Qt::UserRole + 1, column);
      m_table->setItem(i, column, item);
    }
  }

  auto* label = new QTableWidgetItem;
  label->setData(Qt::UserRole, -1);

  m_table->setItem(size, 0, label);

  for (int i = 1; i < NUM_COLUMNS; i++)
  {
    auto* no_edit = new QTableWidgetItem;
    no_edit->setFlags(Qt::ItemIsEnabled);
    m_table->setItem(size, i, no_edit);
  }

  m_updating = false;
}

void WatchWidget::closeEvent(QCloseEvent*)
{
  Settings::Instance().SetWatchVisible(false);
}

void WatchWidget::showEvent(QShowEvent* event)
{
  UpdateButtonsEnabled();
  Update();
}

void WatchWidget::OnDelete()
{
  if (m_table->selectedItems().empty())
    return;

  DeleteSelectedWatches();
}

void WatchWidget::OnClear()
{
  PowerPC::debug_interface.ClearWatches();
  Update();
}

void WatchWidget::OnNewWatch()
{
  const QString text =
      QInputDialog::getText(this, tr("Input"), tr("Enter address to watch:"), QLineEdit::Normal,
                            QString{}, nullptr, Qt::WindowCloseButtonHint);
  bool good;
  const uint address = text.toUInt(&good, 16);

  if (!good)
  {
    ModalMessageBox::warning(this, tr("Error"), tr("Invalid watch address: %1").arg(text));
    return;
  }

  const QString name = QStringLiteral("mem_%1").arg(address, 8, 16, QLatin1Char('0'));
  AddWatch(name, address);
}

void WatchWidget::OnLoad()
{
  IniFile ini;

  std::vector<std::string> watches;

  if (!ini.Load(File::GetUserPath(D_GAMESETTINGS_IDX) + SConfig::GetInstance().GetGameID() + ".ini",
                false))
  {
    return;
  }

  if (ini.GetLines("Watches", &watches, false))
  {
    for (const auto& watch : PowerPC::debug_interface.GetWatches())
    {
      PowerPC::debug_interface.UnsetPatch(watch.address);
    }
    PowerPC::debug_interface.ClearWatches();
    PowerPC::debug_interface.LoadWatchesFromStrings(watches);
  }

  Update();
}

void WatchWidget::OnSave()
{
  IniFile ini;
  ini.Load(File::GetUserPath(D_GAMESETTINGS_IDX) + SConfig::GetInstance().GetGameID() + ".ini",
           false);
  ini.SetLines("Watches", PowerPC::debug_interface.SaveWatchesToStrings());
  ini.Save(File::GetUserPath(D_GAMESETTINGS_IDX) + SConfig::GetInstance().GetGameID() + ".ini");
}

void WatchWidget::ShowContextMenu()
{
  QMenu* menu = new QMenu(this);

  if (!m_table->selectedItems().empty())
  {
    const std::size_t count = m_table->selectionModel()->selectedRows().count();
    if (count > 1)
    {
      // i18n: This kind of "watch" is used for watching emulated memory.
      // It's not related to timekeeping devices.
      menu->addAction(tr("&Delete Watches"), this, [this] { DeleteSelectedWatches(); });
      // i18n: This kind of "watch" is used for watching emulated memory.
      // It's not related to timekeeping devices.
      menu->addAction(tr("&Lock Watches"), this, [this] { LockSelectedWatches(); });
      // i18n: This kind of "watch" is used for watching emulated memory.
      // It's not related to timekeeping devices.
      menu->addAction(tr("&Unlock Watches"), this, [this] { UnlockSelectedWatches(); });
    }
    else if (count == 1)
    {
      auto row_variant = m_table->selectedItems()[0]->data(Qt::UserRole);

      if (!row_variant.isNull())
      {
        int row = row_variant.toInt();

        if (row >= 0)
        {
          menu->addAction(tr("Show in Memory"), this, [this, row] { ShowInMemory(row); });
          // i18n: This kind of "watch" is used for watching emulated memory.
          // It's not related to timekeeping devices.
          menu->addAction(tr("&Delete Watch"), this, [this, row] { DeleteWatchAndUpdate(row); });
          menu->addAction(tr("&Add Memory Breakpoint"), this,
                          [this, row] { AddWatchBreakpoint(row); });
        }
      }
    }
  }

  menu->addSeparator();

  menu->addAction(tr("Update"), this, &WatchWidget::Update);

  menu->exec(QCursor::pos());
}

void WatchWidget::OnItemChanged(QTableWidgetItem* item)
{
  if (m_updating || item->data(Qt::UserRole).isNull())
    return;

  int row = item->data(Qt::UserRole).toInt();
  int column = item->data(Qt::UserRole + 1).toInt();

  if (row == -1)
  {
    if (!item->text().isEmpty())
    {
      AddWatch(item->text(), 0);

      Update();
      return;
    }
  }
  else
  {
    switch (column)
    {
    case COLUMN_INDEX_LABEL:
      if (item->text().isEmpty())
        DeleteWatchAndUpdate(row);
      else
        PowerPC::debug_interface.UpdateWatchName(row, item->text().toStdString());
      break;
    case COLUMN_INDEX_ADDRESS:
    case COLUMN_INDEX_HEX:
    case COLUMN_INDEX_DECIMAL:
    {
      bool good;
      const bool column_uses_hex_formatting =
          column == COLUMN_INDEX_ADDRESS || column == COLUMN_INDEX_HEX;
      quint32 value = item->text().toUInt(&good, column_uses_hex_formatting ? 16 : 10);

      if (good)
      {
        if (column == COLUMN_INDEX_ADDRESS)
        {
          const auto& watch = PowerPC::debug_interface.GetWatch(row);
          PowerPC::debug_interface.UnsetPatch(watch.address);
          PowerPC::debug_interface.UpdateWatchAddress(row, value);
          if (watch.locked)
            LockWatchAddress(value);
        }
        else
        {
          PowerPC::HostWrite_U32(value, PowerPC::debug_interface.GetWatch(row).address);
        }
      }
      else
      {
        ModalMessageBox::critical(this, tr("Error"), tr("Invalid input provided"));
      }
      break;
    }
    case COLUMN_INDEX_LOCK:
    {
      PowerPC::debug_interface.UpdateWatchLockedState(row, item->checkState() == Qt::Checked);
      const auto& watch = PowerPC::debug_interface.GetWatch(row);
      if (watch.locked)
        LockWatchAddress(watch.address);
      else
        PowerPC::debug_interface.UnsetPatch(watch.address);
      break;
    }
    }

    Update();
  }
}

void WatchWidget::LockWatchAddress(u32 address)
{
  const std::string memory_data_as_string = PowerPC::HostGetString(address, 4);

  std::vector<u8> bytes;
  for (const char c : memory_data_as_string)
  {
    bytes.push_back(static_cast<u8>(c));
  }

  PowerPC::debug_interface.SetFramePatch(address, bytes);
}

void WatchWidget::DeleteSelectedWatches()
{
  std::vector<int> row_indices;
  for (const auto& index : m_table->selectionModel()->selectedRows())
  {
    const auto* item = m_table->item(index.row(), index.column());
    const auto row_variant = item->data(Qt::UserRole);
    if (row_variant.isNull())
      continue;

    row_indices.push_back(row_variant.toInt());
  }

  // Sort greatest to smallest, so we
  // don't stomp on existing indices
  std::sort(row_indices.begin(), row_indices.end(), std::greater{});
  for (const int row : row_indices)
  {
    DeleteWatch(row);
  }

  Update();
}

void WatchWidget::DeleteWatch(int row)
{
  PowerPC::debug_interface.UnsetPatch(PowerPC::debug_interface.GetWatch(row).address);
  PowerPC::debug_interface.RemoveWatch(row);
}

void WatchWidget::DeleteWatchAndUpdate(int row)
{
  DeleteWatch(row);
  Update();
}

void WatchWidget::AddWatchBreakpoint(int row)
{
  emit RequestMemoryBreakpoint(PowerPC::debug_interface.GetWatch(row).address);
}

void WatchWidget::ShowInMemory(int row)
{
  emit ShowMemory(PowerPC::debug_interface.GetWatch(row).address);
}

void WatchWidget::AddWatch(QString name, u32 addr)
{
  PowerPC::debug_interface.SetWatch(addr, name.toStdString());
  Update();
}

void WatchWidget::LockSelectedWatches()
{
  for (const auto& index : m_table->selectionModel()->selectedRows())
  {
    const auto* item = m_table->item(index.row(), index.column());
    const auto row_variant = item->data(Qt::UserRole);
    if (row_variant.isNull())
      continue;
    const int row = row_variant.toInt();
    const auto& watch = PowerPC::debug_interface.GetWatch(row);
    if (watch.locked)
      continue;
    PowerPC::debug_interface.UpdateWatchLockedState(row, true);
    LockWatchAddress(watch.address);
  }

  Update();
}

void WatchWidget::UnlockSelectedWatches()
{
  for (const auto& index : m_table->selectionModel()->selectedRows())
  {
    const auto* item = m_table->item(index.row(), index.column());
    const auto row_variant = item->data(Qt::UserRole);
    if (row_variant.isNull())
      continue;
    const int row = row_variant.toInt();
    const auto& watch = PowerPC::debug_interface.GetWatch(row);
    if (!watch.locked)
      continue;
    PowerPC::debug_interface.UpdateWatchLockedState(row, false);
    PowerPC::debug_interface.UnsetPatch(watch.address);
  }

  Update();
}
