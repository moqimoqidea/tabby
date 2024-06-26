package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.options.ShowSettingsUtil
import com.tabbyml.intellijtabby.settings.Configurable

class OpenSettings : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    ShowSettingsUtil.getInstance().showSettingsDialog(e.project, Configurable::class.java)
  }
}