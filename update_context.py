#!/usr/bin/env python3
"""
Context Update Utility
Automatically updates context.txt when significant changes occur
"""

import os
import datetime
from pathlib import Path

def update_context(new_command=None, new_action=None, status_update=None):
    """Update context.txt with new information"""
    
    context_file = Path("context.txt")
    if not context_file.exists():
        return
    
    # Read current context
    with open(context_file, 'r') as f:
        content = f.read()
    
    # Update timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    content = content.replace(
        "# Last Updated: 2025-07-19",
        f"# Last Updated: {current_time}"
    )
    
    # Add new command/action if provided
    if new_command and new_action:
        command_section = "## COMMAND HISTORY & DECISIONS"
        
        # Find the end of command history section
        lines = content.split('\n')
        insert_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith("## CURRENT PROJECT STRUCTURE"):
                insert_idx = i
                break
        
        if insert_idx:
            # Create new entry
            entry_num = len([l for l in lines if l.strip().startswith(f"{i+1}.")]) + 1
            new_entry = f"\n{entry_num}. **{new_command}**\n   - Action: {new_action}\n"
            
            lines.insert(insert_idx, new_entry)
            content = '\n'.join(lines)
    
    # Update status if provided
    if status_update:
        # Find and update status section
        status_section = "## TECHNICAL IMPLEMENTATION STATUS"
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith(status_section):
                # Add new status after existing ones
                j = i + 1
                while j < len(lines) and (lines[j].startswith('✅') or lines[j].startswith('❌') or lines[j].strip() == ''):
                    j += 1
                lines.insert(j, f"✅ {status_update}")
                break
        
        content = '\n'.join(lines)
    
    # Write back
    with open(context_file, 'w') as f:
        f.write(content)
    
    print(f"[CONTEXT] Updated context.txt - {current_time}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        command = sys.argv[1]
        action = sys.argv[2]
        status = sys.argv[3] if len(sys.argv) > 3 else None
        update_context(command, action, status)
    else:
        print("Usage: python update_context.py 'command' 'action' ['status']")