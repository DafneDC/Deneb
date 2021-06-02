from os import replace
import ply.lex as lex
import ply.yacc as yacc
import sys
import collections
from collections import deque

#Variables auxiliares para la tabla de simbolos
TablaSimbolos = {}
var = deque()
i = 0
valores = [0 for i in range (1000)]

#Variables auxiliares para código intermedio
operandos = []
temporal = ['T49','T48','T47','T46','T45','T44','T43','T42','T41','T40','T39','T38','T37','T36','T35','T34',
            'T33','T32','T31','T30','T29','T28','T27','T26','T25','T24','T23','T22','T21','T20','T19','T18',
            'T17','T16','T15','T14','T13','T12','T11','T10','T9','T8','T7','T6','T5','T4','T3','T2','T1']
cuadruple = []

#variables auxiliares para los ciclos
Contador = 0
saltos = deque()

def rellenar(dir, val):
    if(cuadruple[dir][0] == 'goto'):
        cuadruple[dir][1] = val
    else:
        cuadruple[dir][2] = str(val)

#variables para el procedure
#etiquetas = []
procedure = {}

#variables para los arrays
datos_array = []
line2 = []
dimensiones = deque()
array_eje = deque()
line_pos = deque()
M = 1

#variables para la ejecución
PC = 0

def is_in_array(nombre):
    for x in range(len(datos_array)):
        line_array = datos_array[x]
        if (line_array[0] == nombre):
            return True
    
    return False

def ejecutor():
    global PC
    pila_ejecutador = deque()

    while (True):

        inst = cuadruple[PC]
        #print("inst:", inst)

        if (inst[0] == 'goto'):
            PC = int(inst[1])

        elif (inst[0] == 'gotoF'):
            op = list(TablaSimbolos.keys()).index(inst[1])
            if (valores[op] == False):
                PC = int(inst[2])
            else:
                PC += 1

        elif (inst[0] == 'goto_P'):
            pila_ejecutador.append(PC+1)
            PC = inst[1]

        elif (inst[0] == 'endproc'):
            PC = pila_ejecutador.pop()

        elif(inst[0] == 'endall'):
            break

        elif (inst[0] == 'input'):
            line2 = []
            index = list(TablaSimbolos.keys()).index(inst[1])

            valores[index] = int(input())

            if (array_eje):
                line2.append(inst[0])
                line2.append(array_eje.pop())
                cuadruple[PC] = line2

            PC +=1

        elif (inst[0] == 'print'):
            line2 = []
            if(inst[1] in TablaSimbolos):
                op = valores[list(TablaSimbolos.keys()).index(inst[1])]
            elif(type(inst[1]) == int or type(inst[1]) == float or type(inst[1]) == str): 
                op = inst[1]
            else:
                inicio = 0
                helper = inst[1]
                op_name = deque()
                for x in range(len(helper)):
                    if(helper[x] == ' '):
                        op_name.appendleft(helper[inicio:x])
                        inicio = x + 1
                
                op_name.appendleft(helper[inicio:len(helper)])
                len_op = len(op_name)
                lol = ''
                for x in range (len_op):
                    a = op_name.pop()
                    if (type(a) == 'int'):
                        lol = lol + str(a)
                    else:
                        lol = lol + a
                b = str(lol)
                if(b in TablaSimbolos):
                    op = valores[list(TablaSimbolos.keys()).index(b)]

            print(op)

            if (array_eje):
                line2.append(inst[0])
                line2.append(array_eje.pop())
                cuadruple[PC] = line2

            PC += 1

        elif (inst[0] == '='):
            line2 = []
            if(inst[1] in TablaSimbolos):
                op2 = valores[list(TablaSimbolos.keys()).index(inst[1])]
            else: 
                op2 = inst[1]

            op1 = list(TablaSimbolos.keys()).index(inst[2])

            valores[op1] = op2
            
            if(array_eje):
                line2.append(inst[0])
                while (array_eje):
                    pos = line_pos.pop()
                    new_op = array_eje.pop()
                    if (pos == 1):
                        op_aux = inst[2]
                        line2.append(new_op)
                        line2.append(op_aux)
                    elif (pos == 2):
                        op_aux = inst[1]
                        line2.append(op_aux)
                        line2.append(new_op)

                cuadruple[PC] = line2

            PC += 1
        
        elif (inst[0] == 'rellenar'):
            var_dim = deque()
            name = ''
            new_name = ''
            new_dir = 0
            inicio = 0
            
            PC += 1
            line = cuadruple[PC]
            if(line [0] == 'rellenar'):
                while(line[0] == 'rellenar'):
                    PC += 1
                    line = cuadruple[PC]
                
            if (len(line) == 2):
                name = line[1]
            elif (len(line) == 3):  #Tipo =
                if( not (line[1] in TablaSimbolos)):
                    name = line[1]
                    line_pos.append(1)
                elif( not (line[2] in TablaSimbolos)):
                    name = line[2]
                    line_pos.append(2)

            elif(len(line) == 4):   #Tipo operando
                if( not (line[1] in TablaSimbolos)):
                    name = line[1]
                    line_pos.append(1)
                elif( not (line[2] in TablaSimbolos)):
                    name = line[2]
                    line_pos.append(2)
                elif( not (line[3] in TablaSimbolos)):
                    name = line[3]
                    line_pos.append(3)

            array_eje.append(name)
            for x in range(len(name)):
                if(name[x] == ' '):
                    var_dim.appendleft(name[inicio:x])
                    inicio = x + 1

            var_dim.appendleft(name[inicio:len(name)])
            
            line_ar = datos_array[0]
            name = var_dim.pop()

            for x in range(len(datos_array)):
                line_array = datos_array[x]
                if (line_array[0] == name):
                    line_ar = datos_array[x]

            len_var = len(var_dim)
            for x in range(len_var):
                vari = var_dim.pop()
                k = line_ar[x+3]
                if(vari in TablaSimbolos):
                    i = valores[list(TablaSimbolos.keys()).index(vari)]
                    new_dir = new_dir + (i*k)
                else:
                    i = vari
                    new_dir = new_dir + (int(i)*k)
                
            new_dir = new_dir + line_ar[1]
            
            #Construcción del cadruplo
            if (len(line) == 2):
                line.pop()
                new_name = name + str(new_dir)
                line.append(new_name)

            elif (len(line) == 3):
                pos = line_pos.pop()

                if (pos == 1):
                    op_aux = line.pop()
                    line.pop()
                    new_name = name + str(new_dir)
                    line.append(new_name)
                    line.append(op_aux)
                elif (pos == 2):
                    line.pop()
                    new_name = name + str(new_dir)
                    line.append(new_name)
                    
                line_pos.append(pos)

            elif (len(line) == 4):
                pos = line_pos.pop()

                if (pos == 1):
                    op_aux = line.pop()
                    op_aux2 = line.pop()
                    line.pop()
                    new_name = name + str(new_dir)
                    line.append(new_name)
                    line.append(op_aux2)
                    line.append(op_aux)
                elif (pos == 2):
                    op_aux = line.pop()
                    line.pop()
                    new_name = name + str(new_dir)
                    line.append(new_name)
                    line.append(op_aux)
                elif (pos == 3):
                    line.pop()
                    new_name = name + str(new_dir)
                    line.append(new_name)
                
                line_pos.append(pos)
                
            cuadruple[PC] = line

        #---------------------------- operando -------------------------------
        elif (len(inst) == 4):
            line2 = []
            tipo_res = 'INT'
            opcode = inst[0]

            if (inst[1] in TablaSimbolos):
                op2 = valores[list(TablaSimbolos.keys()).index(inst[1])]
            else:
                op2 = inst[1]

            if (inst[2] in TablaSimbolos):
                op1 = valores[list(TablaSimbolos.keys()).index(inst[2])]
            else:
                op1 = inst[2]

            res = list(TablaSimbolos.keys()).index(inst[3])
            if (TablaSimbolos.get(inst[3]) == 'INT'):
                tipo_res = 'INT'
            else:
                tipo_res = 'FLOAT'

            if (tipo_res == 'INT'):
                if (opcode == '+'):
                    valores[res] = int(op2 + op1)
                elif (opcode == '-'):
                    valores[res] = int(op2 - op1)
                elif (opcode == '*'):
                    valores[res] = int(op2 * op1)
                elif (opcode == '/'):
                    valores[res] = int(op2 / op1)
            elif(tipo_res == 'FLOAT'):
                if (opcode == '+'):
                    valores[res] = float(op2 + op1)
                elif (opcode == '-'):
                    valores[res] = float(op2 - op1)
                elif (opcode == '*'):
                    valores[res] = float(op2 * op1)
                elif (opcode == '/'):
                    valores[res] = float(op2 / op1)
            
            if (opcode == '=='):
                valores[res] = (op2 == op1)
            elif (opcode == '<='):
                valores[res] = (op2 <= op1)
            elif (opcode == '>='):
                valores[res] = (op2 >= op1)
            elif (opcode == '<'):
                valores[res] = (op2 < op1)
            elif (opcode == '>'):
                valores[res] = (op2 > op1)
            elif (opcode == '<>'):
                valores[res] = (op2 != op1)
            elif (opcode == 'OR'):
                valores[res] = (op2 or op1)
            elif (opcode == 'AND'):
                valores[res] = (op2 and op1)

            if(array_eje):
                line2.append(inst[0])
                while (array_eje):
                    pos = line_pos.pop()
                    new_op = array_eje.pop()
                    if (pos == 1):
                        op_aux = inst[2]
                        op_aux2 = inst[3]
                        line2.append(new_op)
                        line2.append(op_aux)
                        line2.append(op_aux2)
                    elif (pos == 2):
                        op_aux = inst[1]
                        op_aux2 = inst[3]
                        line2.append(op_aux)
                        line2.append(new_op)
                        line2.append(op_aux2)
                    elif(pos == 3):
                        op_aux = inst[1]
                        op_aux2 = inst[2]
                        line2.append(op_aux)
                        line2.append(op_aux2)
                        line2.append(new_op)

                cuadruple[PC] = line2

            PC += 1

        elif (inst[0] == 'findelprograma'):
            print("Fin de ejecución")
            break

#Reserved words
reserved = {
    'PROGRAM' : 'PROGRAM', 'END' : 'END',
    'DIM' : 'DIM', 'AS' : 'AS',
    'PROCEDURE' : 'PROCEDURE', 'RETURN' : 'RETURN',
    'MAIN' : 'MAIN',
    'GOTO' : 'GOTO', 'GOSUB' : 'GOSUB',
    'FOR' : 'FOR', 'TO' : 'TO', 'NEXT' : 'NEXT',
    'LET' : 'LET',
    'WHILE' : 'WHILE', 'WEND' : 'WEND',
    'DO' : 'DO', 'LOOP' : 'LOOP',
    'IF' : 'IF', 'THEN' : 'THEN', 'ELSE' : 'ELSE',
    'INPUT' : 'INPUT', 'PRINT' : 'PRINT',
    'OR' : 'OR', 'AND' : 'AND', 'NOT' : 'NOT',
    'INT' : 'TY_INT', 'FLOAT' : 'TY_FLOAT', 'STRING' : 'TY_STRING'
}

#tokens
tokens = [
    'ID',
    'COMMA',
    'LPAR', 'RPAR',
    'LBRACKET', 'RBRACKET',
    'EQUALS', 
    'SUM', 'SUB', 'MULT', 'DIV',
    'LESSTHAN', 'GREATERTHAN',
    'LESSOREQUAL', 'GREATEROREQUAL',
    'ISEQUAL', 'ISNOTEQUAL', 
    'INT', 'FLOAT', 'STRING'
] + list(reserved.values())

#Values of the tokens
t_COMMA = r','
t_LPAR = r'\('
t_RPAR = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_EQUALS = r'\='
t_SUM = r'\+'
t_SUB = r'\-'
t_MULT = r'\*'
t_DIV = r'\/'
t_LESSTHAN = r'\<'
t_GREATERTHAN = r'\>'
t_LESSOREQUAL = r'\<\='
t_GREATEROREQUAL = r'\>\='
t_ISEQUAL = r'\=\='
t_ISNOTEQUAL = r'\<\>'

# Ignored characters (spaces y tabs)
t_ignore = ' \t'

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_STRING(t):
    r'\"[a-zA-Z 0-9:!@#$%^&*()-+=/?<>,\\]+\"'
    t.value = str(t.value)
    return t

def t_error(t):
    # print(f"Illegal character {t.value[0]} at line {t.lineno}")
    t.lexer.skip(1)

def t_newline(t):
	r'\n+'
	t.lexer.lineno += len(t.value)

#Building the lexer
lexer = lex.lex()

#grammar starts here 
def p_programa(p):
    '''
    programa : aux_pr variable process aux_main block END
    ''' 
    global Contador
    end = ['findelprograma']
    cuadruple.append(end)
    print("CORRECTO")

def p_aux_pr(p):
    '''
    aux_pr : PROGRAM
    '''
    global Contador
    init = ['goto', None]
    cuadruple.append(init)
    Contador += 1

def p_aux_main(p):
    '''
    aux_main : MAIN
    '''
    global Contador
    rellenar(0, Contador)

def p_variable(p):
    '''
    variable : DIM repeat_id AS type repeat_size end_var variable
             | empty
    '''

def p_repeat_id(p):
    '''
    repeat_id : ID COMMA repeat_id
              | ID
    '''  
    if(p[1] in TablaSimbolos or p[1] in var): 
        print("La variable: '", p[1], "' ya esta definida :s")
    else:
        var.append(p[1])

def p_type(p):
    '''
    type : TY_INT
         | TY_FLOAT
         | TY_STRING
    '''
    global var
    
    for x in var:
        TablaSimbolos.update({x:p[1]})

def p_repeat_size(p):
    '''
    repeat_size : LBRACKET INT RBRACKET repeat_size
                | empty
    '''
    global M
    if(len(p) > 2):
        M *= p[2]
        dimensiones.appendleft(p[2])
        
def p_end_var(p):
    '''
    end_var : empty
    '''
    global M
    
    if(dimensiones):
        for x in var:
            ty = TablaSimbolos.get(x)

            #Datos del arreglo/matriz/cubo
            auxi = [x, len(TablaSimbolos), M]
            for dim in dimensiones:
                m_aux = auxi.pop()
                auxi.append(m_aux)
                auxi.append(int(m_aux/dim))

            datos_array.append(auxi)
            #print("dimensiones:",dimensiones)
            for y in range(M):
                name = x + str(len(TablaSimbolos))
                TablaSimbolos.update({name:ty})

    #limpiado de variables
    dimensiones.clear()
    length = len(var)
    for x in range(length):
        var.pop()
    M = 1

#MAIN ---------------------------------------------------------------------------------------------------------
def p_block(p):
    '''
    block : repeat_state
    '''

def p_repeat_state(p):
    '''
    repeat_state : statement repeat_state
                 | empty
    '''

#Procedure ---------------------------------------------------------------------------------------------------
def p_process(p):
    '''
    process : PROCEDURE aux_id variable block fin_proc process
            | empty
    '''

def p_aux_id(p):
    '''
    aux_id : ID
    '''
    global Contador
    procedure.update({p[1]:Contador})

def p_fin_proc(p):
    '''
    fin_proc : RETURN
    '''
    global Contador
    end = ['endproc']
    cuadruple.append(end)
    Contador += 1

def p_statement(p):
    '''
    statement : GOTO ID
              | ID
    '''
    global Contador
    if(p[1] == 'endall'):
        bre = ['endall']
        cuadruple.append(bre)
        Contador += 1

def p_statement_procedure(p):
    '''
    statement : GOSUB ID
    '''
    global Contador
    if (p[2] in procedure):
        dir = procedure.get(p[2])
        proc = ['goto_P', dir]
        cuadruple.append(proc)
        Contador += 1
    else:
        print(p[2], "no esta definida :c")
        quit()

# Definicion de let -----------------------------------------------------------------------------------------
def p_statement_let(p):
    '''
    statement : LET var EQUALS expression
    '''
    global Contador
    op2 = operandos.pop()
    op1 = operandos.pop()
    Contador += 1
    quad = ['=',op2,op1]
    cuadruple.append(quad)

# Definición de variables -------------------------------------------------------------------------------------
def p_var(p):
    '''
    var : ID repeat_size_v
    '''
    global Contador
    dir = ''
    aux = []
    switch = False
    if(p[1] in TablaSimbolos.keys()):
        for x in range(len(datos_array)):
            line2 = datos_array[x]
            if (p[1] == line2[0]):
                switch = True
                line = line2

        if(switch == True):
            times = len(line) - 3
            for x in range(times): 
                S = operandos.pop()
                if (S in TablaSimbolos):
                    redir = ['rellenar', Contador+1, S]
                    cuadruple.append(redir)
                    Contador += 1
                    aux.append(S)
                else:
                    aux.append(S)
            
            for x in range(len(aux)):
                S = aux.pop()
                if (S in TablaSimbolos):
                    dir = dir + ' ' + S
                else:
                    dir = dir + ' ' + str(S)

            new_p = p[1] + dir
            operandos.append(new_p)
        else:
            operandos.append(p[1])

    else:
        print(p[1], "no esta declarada :(")
        quit()
    
def p_repeat_size_v(p):
    '''
    repeat_size_v : aux_ar2 
                  | empty
    '''

def p_aux_r2(p):
    '''
    aux_ar2 : LBRACKET expression else_size RBRACKET
    '''

def p_else_size(p):
    '''
    else_size : COMMA expression else_size
              | empty
    '''

# Definición del for ------------------------------------------------------------------------------------------
def p_statement_for(p):
    '''
    statement : FOR aux_F1 expression aux_F2 TO expression aux_F3 repeat_state NEXT aux_F4
    '''

def p_for_aux_F1(p):
    '''
    aux_F1 : ID EQUALS 
    '''
    operandos.append(p[1])

def p_for_aux_F2(p):
    '''
    aux_F2 : empty
    '''
    global Contador
    ope = operandos.pop()
    id = operandos.pop()
    For1 = ['=', ope, id]
    operandos.append(id)
    cuadruple.append(For1)
    Contador += 1

def p_for_aux_F3(p):
    '''
    aux_F3 : empty
    '''
    global Contador

    temp = temporal.pop()
    TablaSimbolos.update({temp:'BOOL'})
    ope = operandos.pop()
    id = operandos.pop()
    For2 = ['<=',id,ope,temp]
    cuadruple.append(For2)
    operandos.append(id)
    Contador += 1

    For3 = ['gotoF',temp,None]
    cuadruple.append(For3)
    Contador += 1

    saltos.append(Contador-2)

def p_for_aux_F4(p):
    '''
    aux_F4 : ID
    '''
    global Contador
    id = operandos.pop()
    inc = ['+',id,1,id]
    cuadruple.append(inc)
    Contador += 1

    r = saltos.pop()
    ret = ['goto',r]
    cuadruple.append(ret)
    Contador += 1

    rellenar(r+1, Contador)

# Definición de del ciclo  -------------------------------------------------------------------------------
def p_statement_while(p):
    '''
    statement : WHILE aux_w1 expression aux_w2 DO repeat_state WEND aux_w3
    '''

def p_aux_w1(p):
    '''
    aux_w1 : empty
    '''
    global Contador
    saltos.append(Contador)

def p_aux_w2(p):
    '''
    aux_w2 : empty
    '''
    global Contador
    ope = operandos.pop()
    gotoF = ['gotoF',ope,None]
    cuadruple.append(gotoF)
    Contador += 1
    saltos.append(Contador - 1)

def p_aux_w3(p):
    '''
    aux_w3 : empty
    '''
    global Contador
    f = saltos.pop()
    r = saltos.pop()
    goto = ['goto', r]
    cuadruple.append(goto)
    Contador += 1
    rellenar(f, Contador)

# Definición de del ciclo DO ----------------------------------------------------------------------------------
def p_statement_do(p):
    '''
    statement : DO aux_d1 repeat_state LOOP WHILE expression aux_d2
    '''
    
def p_aux_d1(p):
    '''
    aux_d1 : empty
    '''
    global Contador
    saltos.append(Contador)

def p_aux_d2(p):
    '''
    aux_d2 : empty
    '''
    ope = operandos.pop()
    dir = saltos.pop()
    gotoF = ['gotoF', ope, dir]
    cuadruple.append(gotoF)

# Definición de del if ----------------------------------------------------------------------------------------

def p_statement_if(p):
    '''
    statement : IF expression aux_if1 THEN repeat_state ELSE aux_if2 repeat_state END IF aux_if3
    '''
    #pass

def p_aux_if1(p):
    '''
    aux_if1 : empty
    '''
    global Contador
    ope = operandos.pop()
    gotoF = ['gotoF',ope,None]
    cuadruple.append(gotoF)
    Contador += 1
    saltos.append(Contador - 1)

def p_aux_if2(p):
    '''
    aux_if2 : empty
    '''
    global Contador
    goto = ['goto', None]
    cuadruple.append(goto)
    Contador += 1
    sal = saltos.pop()
    rellenar(sal, Contador)
    saltos.append(Contador - 1)

def p_aux_if3(p):
    '''
    aux_if3 : empty
    '''
    global Contador
    fin = saltos.pop()
    rellenar(fin, Contador)

# Definición de del input -------------------------------------------------------------------------------------
def p_statement_input(p):
    '''
    statement : INPUT repeat_elem
    '''
    global Contador
    op = operandos.pop()
    inop = ['input', op]
    cuadruple.append(inop)
    Contador += 1

# Definición de del print -------------------------------------------------------------------------------------
def p_statement_print(p):
    '''
    statement : PRINT repeat_elem
    '''
    global Contador
    op = operandos.pop()
    priop = ['print', op]
    cuadruple.append(priop)
    Contador += 1

def p_repeat_elem(p):
    '''
    repeat_elem : elem COMMA repeat_elem
                | elem
    '''

def p_elem(p):
    '''
    elem : STRING
         | FLOAT
         | INT
         | var
    '''
    if p[1] != None:
        operandos.append(p[1])
    
def p_expression(p):
    '''
    expression : first_exp GREATERTHAN first_exp
               | first_exp GREATEROREQUAL first_exp
               | first_exp LESSTHAN first_exp
               | first_exp LESSOREQUAL first_exp
               | first_exp ISEQUAL first_exp
               | first_exp ISNOTEQUAL first_exp
               | first_exp
    '''
    global Contador
    if (len(p) == 4):
        op2 = operandos.pop()
        op1 = operandos.pop()
        r = temporal.pop()
        TablaSimbolos.update({r:'BOOL'})
        Contador += 1
        quad = [p[2],op1,op2,r]
        cuadruple.append(quad)
        operandos.append(r)

        
def p_first_exp(p):
    '''
    first_exp : term
              | first_exp SUM term
              | first_exp SUB term
              | first_exp OR term
    '''
    global Contador
    if (len(p) == 4):
        op2 = operandos.pop()
        op1 = operandos.pop()
        r = temporal.pop()

        op_name_p = deque()
        op_name_p2 = deque()
        inicio = 0
        x = 0

        if(not(op1 in TablaSimbolos) and (not(type(op1) == int)) and (not(type(op1) == float))):
            for x in range(len(op1)):
                if(op1[x] == ' '):
                    op_name_p.appendleft(op1[inicio:x])
                    inicio = x + 1
            opc1 = op_name_p.pop()
        else:
            opc1 = op1

        if(not(op2 in TablaSimbolos) and (not(type(op2) == int)) and (not(type(op2) == float))):
            for x in range(len(op2)):
                if(op2[x] == ' '):
                    op_name_p2.appendleft(op2[inicio:x])
                    inicio = x + 1
            opc2 = op_name_p2.pop()
        else:
            opc2 = op2

        if (p[2] == 'OR'):
            TablaSimbolos.update({r:'BOOL'})
        elif ((TablaSimbolos.get(opc2) == 'INT' or type(opc2) == int) and (TablaSimbolos.get(opc1) == 'INT' or type(opc1) == int)):
            TablaSimbolos.update({r:'INT'})
        else:
            TablaSimbolos.update({r:'FLOAT'})
        Contador += 1
        quad = [p[2],op1,op2,r]
        cuadruple.append(quad)
        operandos.append(r)

def p_term(p):
    '''
    term : factor
        | term MULT factor
        | term DIV factor
        | term AND factor
    '''
    global Contador
    if (len(p) == 4):
        op2 = operandos.pop()
        op1 = operandos.pop()
        r = temporal.pop()

        op_name_p = deque()
        op_name_p2 = deque()
        inicio = 0
        x = 0

        if(not(op1 in TablaSimbolos) and (not(type(op1) == int)) and (not(type(op1) == float))):
            for x in range(len(op1)):
                if(op1[x] == ' '):
                    op_name_p.appendleft(op1[inicio:x])
                    inicio = x + 1
            opc1 = op_name_p.pop()
        else:
            opc1 = op1

        if(not(op2 in TablaSimbolos) and (not(type(op2) == int)) and (not(type(op2) == float))):
            for x in range(len(op2)):
                if(op2[x] == ' '):
                    op_name_p2.appendleft(op2[inicio:x])
                    inicio = x + 1
            opc2 = op_name_p2.pop()
        else:
            opc2 = op2

        if (p[2] == 'AND'):
            TablaSimbolos.update({r:'BOOL'})
        elif ((TablaSimbolos.get(opc2) == 'INT' or type(opc2) == int) and (TablaSimbolos.get(opc1) == 'INT' or type(opc1) == int)):
            TablaSimbolos.update({r:'INT'})
        else:
            TablaSimbolos.update({r:'FLOAT'})
        Contador += 1
        quad = [p[2],op1,op2,r]
        cuadruple.append(quad)
        operandos.append(r)

def p_factor(p):
    '''
    factor : other_fact
           | NOT other_fact
    '''
    #if (len(p) == 3):
    #    op2 = operandos.pop()
    #    r = temporal.pop()
    #    if (p[2] == 'NOT'):
    #        TablaSimbolos.update({r:'BOOL'})
    #    else:
    #        TablaSimbolos.update({r:'INT'})
    #    quad = [p[1],op2,r]
    #    cuadruple.append(quad)

def p_other_fact(p):
    '''
    other_fact : elem
               | LPAR expression RPAR
    '''

def p_error(p):
    print("\tINCORRECTO")
    #quit()

def p_empty(p):
    '''
    empty : 
    '''

# Build the parser
parser = yacc.yacc()

#Se idica cual es el código a ejecutar
with open('code10.txt', 'r') as file:
    data = file.read()    
    parser.parse(data)

#Print tabla de simbolos
#print("\nTabla de simbolos:")
#print("Tamaño de la Tabla de simbolos:", len(TablaSimbolos))
#for value in TablaSimbolos.items():
#	print(i, value)
#	i += 1

#print los datos de los arreglos
#print("\nDatos de los arrays")
#print(datos_array)

#print cuadruplos 
#print("\nCuadruplo:")
#print(cuadruple)

#print la ejecución
print("\nEjecución:")
ejecutor()

#print memoria
#print("\nMemoria:")
#print(valores)

#print la lista de los procedures del programa
#print("\nLista de procedures:")
#print(procedure)