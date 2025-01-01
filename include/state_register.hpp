#ifndef _STATE_REGISTER_H_
#define _STATE_REGISTER_H_


#include <variant>
#include <string>
#include <vector>
#include <map>


using std::vector;
using std::string;
using std::variant;
using std::map;


// (id, label, size)
typedef map<string, variant<int, string>> Info_t;


class StateRegister {
public:
    vector<string> _syms;
    map<string, Info_t> matrix_info;
    map<string, Info_t> data;

    StateRegister() = default;
    ~StateRegister() = default;

    void set(vector<string> vars) {
        for (int id=0; id<vars.size(); ++id) {
            string var = vars[id]; 
            Info_t info = {{"id", id}, {"var", var}, {"size", 1}};
        }
    }
};


#endif
