"""Capture the unmodified color and depth while the effects stay active."""
import lldb

def capture(debugger,command,result,internal):
    target=debugger.GetSelectedTarget()
    symbols=target.FindSymbols('ProcessFrame',lldb.eSymbolTypeCode)
    addresses=[]
    for index in range(symbols.GetSize()):
        context=symbols.GetContextAtIndex(index)
        if context.GetModule().GetFileSpec().GetFilename()=='libSMOCinematic.dylib':
            addresses.append(context.GetSymbol().GetStartAddress().GetLoadAddress(target))
    if len(addresses)!=1:
        result.SetError('Expected one active frame processor. Found: '+str(addresses)); return
    output=lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand('expr ((void(*)(void*))SMOCaptureBugFrame)((void*)0x%x)' % addresses[0],output)
    if not output.Succeeded(): result.SetError(output.GetError()); return
    result.AppendMessage('Frame capture armed. The existing effects remain active.')

def __lldb_init_module(debugger,internal):
    debugger.HandleCommand('command script add -f capture_bug_frame.capture capturebugframe')
